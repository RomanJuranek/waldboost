"""
Support for FPGA stuff.
"""

import logging
import numpy as np
from scipy.ndimage import convolve1d
from itertools import count
from queue import Queue
from collections import deque, defaultdict

from .training import DTree, Learner, as_features
from .samples import Pool


def _bank_pattern(shape, block_shape):
    assert len(shape) in [2,3], "Shape must be (H,W) or (H,W,C)"
    if len(shape) == 2:
        shape += (1,)
    b = np.arange(np.prod(block_shape)).reshape(block_shape)
    n = np.ceil(np.array(shape[:2]) / block_shape)
    banks = np.tile(b, n.astype("i").tolist())
    u,v,c = shape
    banks = np.atleast_3d(np.dstack([banks]*c))
    return banks[:u,:v,...]


class PixelBanks:
    def __init__(self, shape, block_shape):
        self.pattern = _bank_pattern(shape, block_shape)

    def bank_pixels(self, bank_ids):
        return np.concatenate( [np.flatnonzero(self.pattern == b) for b in bank_ids] )


class BankScheduler:
    def __init__(self, n_banks=8):
        self.n_banks = n_banks
        self.bank_counter = count()

    def schedule(self, max_depth=2):
        return [ [next(self.bank_counter) % self.n_banks] for _ in range(max_depth)]


def grad_hist_4(image, bias=4):
    """
    Integer version of grad_hist(image, n_bins=4, full=False)
    Input image must be uint8
    Output image is always int16
    """
    assert image.dtype == np.uint8

    H = np.array( [1,2,1], "i2")
    D = np.array( [1,0,-1], "i2")

    im = image.astype("i2")

    gy = convolve1d(convolve1d(im,H,axis=1), D, axis=0) / 4
    gx = convolve1d(convolve1d(im,H,axis=0), D, axis=1) / 4

    chns = np.empty(im.shape + (4,), "i2")

    theta = np.linspace(0, np.pi, 5)
    cs = np.cos(theta[:-1])
    sn = np.sin(theta[:-1])
    for i,(c,s) in enumerate(zip(cs,sn)):
        chns[...,i] = gx*c - gy*s

    return np.fmax(np.abs(chns-bias), 0)


def H(*p):
    tot = sum(p)
    return -sum((_p/tot) * np.log2(_p/tot) for _p in p)


def _fit_threshold(x0,x1,w0,w1,xmin,xmax):
    """ Find threshold to split v0 and v1 using information gain """
    bins = np.arange(xmin,xmax+1)
    p0,_ = np.histogram(x0, bins, weights=w0)
    p1,_ = np.histogram(x1, bins, weights=w1)
    l0 = np.cumsum(p0)
    l1 = np.cumsum(p1)
    l0_tot = l0[-1]
    l1_tot = l1[-1]
    r0 = l0_tot - l0
    r1 = l1_tot - l1
    left_w = (l0+l1) / (l0_tot + l1_tot)
    right_w = (r0+r1) / (l0_tot + l1_tot)
    metric = H(l0_tot, l1_tot) - (left_w*H(l0+1e-4,l1+1e-4) + right_w * H(r0+1e-4,r1+1e-4))
    k = np.argmax(metric)
    return bins[k+1], metric[k]


def _find_split(x, y, w):
    """ Find best split for the data
    Inputs
    ------
    x : ndarray
        Samples (n_samples, n_features)
    y : ndarray
        Label (0, 1)
    w : ndarray
        Sample weights
    """

    x0,w0 = np.ascontiguousarray(x[y==0]).astype("f"), np.ascontiguousarray(w[y==0])
    x1,w1 = np.ascontiguousarray(x[y==1]).astype("f"), np.ascontiguousarray(w[y==1])
    xmin,xmax = x.min(axis=0), x.max(axis=0)

    threshold, metric = zip(*[_fit_threshold(_x0, _x1, w0, w1, _xmin, _xmax) for _x0,_x1,_xmin,_xmax in zip(x0.T, x1.T,xmin,xmax)])

    k = np.argmax(metric)
    return k, threshold[k], metric[k]


class FPGA_DTree:
    """ Decision tree training algorithm for FPGA
    The difference from waldboost.training.DTree (based on sklearn):
    * Features in split nodes can be restricted according to node depth using
      user-defined function. This is required by FPGA in order to avoid
      colisions while evaluating trees in parallel.

    waldboost.training.DTree and waldboost.fpga.DTree can be used interchangably
    though the former has more options due to sklearn implementation.
    """
    @staticmethod
    def fit(X0, W0, X1, W1, max_depth=2, min_samples_leaf=10, allowed_features=None, clip=3, quantizer=32):
        """ Train decision tree

        Inputs
        ------
        X0, X1 : ndarray
            Samples
        W0, W1 : ndarray or None
            Sample weights
        max_depth : scalar int
            Maximal depth of decision tree
        min_samples_leaf : scalar int/float
            TODO
        allowed_features : callable
            TODO

        Outputs
        -------
        tree : waldboost.training.DTree
            Initialized instance of decision tree. tree.apply, tree.predict and
            others can be called.
        """
        shape = X0.shape[1:]

        X = np.concatenate( [as_features(X0), as_features(X1)] )
        Y = np.array([0]*X0.shape[0] + [1]*X1.shape[0])
        W = np.concatenate( [W0, W1] )

        n_samples = W.size
        _W = W.copy()
        _W[Y==0] /= _W[Y==0].sum() * 2
        _W[Y==1] /= _W[Y==1].sum() * 2

        node_id = count()  # Counter of node ids
        sample_queue = Queue()  # Queue holding sample sets that need to be processed
        sample_queue.put( (np.arange(n_samples), 0, next(node_id)) )  # Initial set includes all samples, depth is 0 and id is 0 (from the counter)
        nodes = dict() # Intermediate representation of tree
        while not sample_queue.empty():
            self_samples, depth, self_index = sample_queue.get()
            # Create either leaf or split node
            create_leaf = (depth == max_depth) or (self_samples.size < min_samples_leaf)
            if create_leaf:
                nodes[self_index] = {"samples": self_samples,
                                     "feature": -1, "threshold": -1,
                                     "left": -1, "right": -1}
                # logging.info(f"Leaf node {nodes[self_index]}")
            else:
                _X = X
                if allowed_features is not None:
                    ftrs = allowed_features[depth]
                    _X = X[:,ftrs]
                feature, threshold, _ = _find_split(_X[self_samples], Y[self_samples], _W[self_samples])
                if allowed_features is not None:
                    ftrs = allowed_features[depth]
                    feature = ftrs[feature]
                # TODO: check for feature/metric feasibility
                # split data
                bin = X[self_samples, feature] <= threshold
                # schedule l/r nodes
                left_index = next(node_id)
                sample_queue.put( (self_samples[ bin], depth+1, left_index) )
                right_index = next(node_id)
                sample_queue.put( (self_samples[~bin], depth+1, right_index) )
                # create node
                nodes[self_index] = {"samples": self_samples,
                                     "feature": feature, "threshold": threshold,
                                     "left": left_index, "right": right_index}

        # Compose final structude
        n_nodes = len(nodes)
        feature = [None] * n_nodes
        threshold = np.empty(n_nodes)
        left = np.empty(n_nodes, "i")
        right = np.empty(n_nodes, "i")
        pred = np.empty(n_nodes, "f")

        for node_idx, node_data in nodes.items():
            f = node_data["feature"]
            feature[node_idx] = np.unravel_index(f, shape) if f>=0 else None
            threshold[node_idx] = node_data["threshold"]
            left[node_idx] = node_data["left"]
            right[node_idx] = node_data["right"]
            idx = node_data["samples"]
            y, w = Y[idx], W[idx]
            w0 = w[y==0].sum() + 1e-3
            w1 = w[y==1].sum() + 1e-3
            pred[node_idx] = np.log(w1 / w0) / 2

        if clip is not None:
            pred = np.clip(pred, -clip, clip)

        if quantizer is not None:
            pred = np.round(quantizer*pred) / quantizer

        # Return initialized waldboost.training.DTree instance
        tree = DTree(feature, threshold, left, right, pred)



        return tree


channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 0,
    "target_dtype": np.int16,
    "channels": [ grad_hist_4 ],
    }


def train(model,
          training_images,
          length=64,
          alpha=0.2,
          max_tp_dist=0.25,
          n_pos=1000,
          n_neg=1000,
          max_depth=2,
          target_p0=1e-5,
          bank_pattern_shape=(2,2),
          clip=3,
          quantizer=32,
          callbacks=[],
          logger=None):
    """ Train new model with FPGA friendly feature access patterns

    The function mimics behaviour of waldboost.train.

    Inputs
    ------
    Inputs have exactly same meaning as in waldboost.train
    bank_pattern_shape : tuple or None
        Size of pattern for bank scheduling. E.g. (2,2) means that there will
        be 4 banks in 2x2 block.
    quantizer : int
        Model response multiplier. The responses and rejection thresholds are
        still trained as floats, but when multiplied with quantizer value,
        they are integers. So the model (after transformation) can be evaluated
        with integer arithmentic.

    Outputs
    -------
    stats : dict
        Same as in waldboost.train

    See also
    --------
    waldboost.train     Basline training pipeline

    """

    def theta(n, p0):
        assert target_p0 > 0, "target_p0 must be positive"
        if 0 < target_p0 < 1:
            theta = None if p0 > target_p0 else -np.inf
        else:
            theta = None if n < target_p0 else -np.inf
        return theta

    logger = logger or logging.getLogger("WaldBoost/FPGA")

    learner = Learner(alpha=alpha, wh=FPGA_DTree, max_depth=max_depth, clip=clip, quantizer=quantizer)
    pool = Pool(model.shape, min_tp=n_pos, min_fp=n_neg)
    pool.max_tp_dist = max_tp_dist

    if bank_pattern_shape is not None:
        banks = PixelBanks(model.shape, bank_pattern_shape)
        scheduler = BankScheduler(np.prod(bank_pattern_shape))

    stats = defaultdict(list)

    for stage in range(1,length+1):
        logger.info(f"Training stage {stage}")
        pool.update(model, training_images)
        X0,H0 = pool.gather_samples(0)
        X1,H1 = pool.gather_samples(1)
        if bank_pattern_shape is not None:
            stage_banks = scheduler.schedule(max_depth)
            ftrs = [banks.bank_pixels(b) for b in stage_banks]
        else:
            ftrs = None
        loss,p0,p1 = learner.fit_stage(model, X0, H0, X1, H1, allowed_features=ftrs, theta=theta(stage,learner.P0))
        for cb in callbacks:
            cb(model, learner, stage)
        stats["loss"].append(loss)
        stats["p0"].append(p0)
        stats["p1"].append(p1)

    return stats
