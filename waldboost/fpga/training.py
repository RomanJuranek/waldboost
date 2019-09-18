import logging
from collections import defaultdict, deque
from itertools import count
from queue import Queue

import numpy as np
from scipy.ndimage import convolve1d

from waldboost.samples import SamplePool
from waldboost.training import Learner, BasicRejectionSchedule, as_features, DTree as BaseDTree

from .banks import BankScheduler, PixelBanks


def H(*p):
    tot = sum(p)
    return -sum((_p/tot) * np.log2(_p/tot) for _p in p)


def _fit_threshold(x0,x1,w0,w1,xmin,xmax):
    """ Find threshold to split v0 and v1 using information gain """
    bins = np.arange(xmin-1,xmax+2)
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


class DTree:
    """ Decision tree training algorithm for FPGA
    The difference from waldboost.training.DTree (based on sklearn):
    * Features in split nodes can be restricted according to node depth using
      user-defined function. This is required by FPGA in order to avoid
      colisions while evaluating trees in parallel.

    waldboost.training.DTree and waldboost.fpga.DTree can be used interchangably
    though the former has more options due to sklearn implementation.
    """
    @staticmethod
    def fit(X0, W0, X1, W1,
            max_depth=2,
            min_samples_leaf=10,
            allowed_features=None,
            clip=3,
            quantizer=32):
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
        return  BaseDTree(feature, threshold, left, right, pred)


def train(model,
          training_images,
          learner=None,
          pool=None,
          length=64,
          max_depth=2,
          theta_schedule=BasicRejectionSchedule(),
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

    Notes
    -----
    When continuing training (initialized model and learner instances are given), bank_pattern_shape
    and quantization parameters should be set to the same values as in the prior training. This is not
    checked anywhere. Using different parameters is allowed but such detector will probably be useless
    in hardware implementation (which is purpose of this function).

    """
    logger = logger or logging.getLogger("WaldBoost/FPGA")

    if len(model) >= length:
        return

    learner = learner or Learner(wh=DTree)

    if learner.wh is not DTree:
        raise ValueError(f"Learner.wh should be waldboost.fpga.DTree")

    if len(model) != len(learner):
        raise RuntimeError("Model length and learner length are not consistent")

    if learner.wh is not DTree:
        logger.warning("Correcting learner.wh to fpga.DTree")
        learner.wh = DTree

    if "max_depth" not in learner.wh_args or \
       learner.wh_args["max_depth"] != max_depth:
        learner.wh_args["max_depth"] = max_depth  # FIXME: we are not supposed to touch Learner internals

    if len(model) > 0:
        logger.info(f"{len(model)} stages are already present, continuing")

    if bank_pattern_shape is not None:
        banks = PixelBanks(model.shape, bank_pattern_shape)
        scheduler = BankScheduler(np.prod(bank_pattern_shape))

    pool = pool or SamplePool()

    for stage in range(len(model), length):
        logger.info(f"Training stage {stage}")
        pool.update(model, training_images)
        X0,H0 = pool.get_false_positives()
        X1,H1 = pool.get_true_positives()
        if bank_pattern_shape is not None:
            stage_banks = scheduler.schedule(max_depth)
            ftrs = [banks.bank_pixels(b) for b in stage_banks]
        else:
            ftrs = None
        loss,p0,p1 = learner.fit_stage(model, X0, H0, X1, H1, allowed_features=ftrs, theta=theta_schedule(stage, learner.false_positive_rate))
        logger.debug(f"Loss: {loss:0.5f}, fpr: {p0:0.5f}, tpr: {p1:0.5f}")
        for cb in callbacks:
            cb(model, learner, stage)

    return learner
