""" WaldBoost training algortithm

Support for training of waldboost classifiers. The functions fit_stage accepts
training data (negative and positive samples X, their responses H and priors P)
and returns new weak classifier of specified type. The module implements
basic DecisionStump and decision tree classes.

Training data are represented as numpy matrices with shapes

X0,X1   (F,N0), (F,N1)
H0,H1   (N0,), (N1,)
P0,P1   scalars 0 < P < 1

Where F is number of features, N0, N1 amout of negative and positive samples.

See also:
waldboost.training.DecisionStump
waldboost.training.DecisionTree
waldboost.training.fit_stage
waldboost.training.fit_model
"""


import logging
import numpy as np
from .samples import SamplePool


logger = logging.getLogger(__name__)


def image_to_features(X):
    """
    Transform X with shape (N,H,W,C) to (H*W*C,N) used for training
    """
    n = X.shape[0]
    return np.moveaxis(X,0,-1).reshape(-1,n)


def find_threshold(f0, w0, f1, w1):
    """
    Find threshold to separate two distributions
    """
    f = np.concatenate([f0,f1])
    fmin = np.min(f)
    fmax = np.max(f)
    # logger.debug(f"f: {f.size}, {f.dtype}, <{fmin},{fmax}>")
    if issubclass(f.dtype.type, np.integer):
        edges = np.r_[fmin-1:fmax+2]
    else:
        assert issubclass(f.dtype.type, np.floating)
        edges = np.linspace(fmin, fmax+1e-3, 256)
    # logger.debug(f"edges: {edges}")
    w0 = w0 / w0.sum()
    w1 = w1 / w1.sum()
    p0,_ = np.histogram(f0, bins=edges, weights=w0)
    p1,_ = np.histogram(f1, bins=edges, weights=w1)
    cdf0 = np.cumsum(p0)
    cdf1 = np.cumsum(p1)
    err = 2 * np.sqrt(cdf0*cdf1 + (1-cdf0)*(1-cdf1))
    k = np.argmin(err)
    return edges[k+1]


def fit_decision_stump(f0, w0, f1, w1, eps=1e-4, min_split_ratio=0.05):
    thr = find_threshold(f0, w0, f1, w1)
    wsum0,_ = np.histogram((f0<thr).astype(np.uint8), bins=[0,1,2], weights=w0)
    wsum1,_ = np.histogram((f1<thr).astype(np.uint8), bins=[0,1,2], weights=w1)
    hs = 0.5 * np.log((wsum1+eps) / (wsum0+eps))
    Z = 2 * np.sqrt(wsum0 * wsum1).sum()
    if min_split_ratio is not None:
        min_n = (f0.size + f1.size) * min_split_ratio
        if ((f0<thr).sum() + (f1<thr).sum()) < min_n or ((f0>=thr).sum() + (f1>=thr).sum()) < min_n:
            Z = np.inf # Penalize divergent solutions
    return (thr, hs), Z


class DStump:
    def __init__(self, shape):
        self.shape = shape
        self.ftr_idx = None
        self.ftr = None
        self.thr = None
        self.predictions = None
        self.err = None
    def set_params(self, ftr_idx, thr, predictions, err):
        self.ftr_idx = ftr_idx
        self.ftr = np.unravel_index(self.ftr_idx, self.shape)
        self.thr = thr
        self.predictions = predictions
        self.err = err
    def fit(self, X0, W0, X1, W1, quantizer=None):
        #print("Q:", quantizer)
        w0,w1 = normalized_weights(W0, W1)
        for ftr_idx,(x0,x1) in enumerate(zip(X0,X1)):
            (thr, hs), err = fit_decision_stump(x0,w0,x1,w1)
            if self.err is None or err < self.err:
                #logging.debug(err)
                self.set_params(ftr_idx, thr, hs, err)
        if quantizer is not None:
            q,q_idx = quantizer(self.predictions)
            #logging.debug(f"{self.predictions} -> {q} ({q_idx})")
            self.predictions = q
            self.predictions_int = q_idx
        return self
    def predict(self, X):
        assert self.is_trained, "Run fit() first"
        return self.predictions[self._predict_bin(X)]
    def predict_on_image(self, X, rs, cs):
        assert self.is_trained, "Run fit() first"
        return self.predictions[self._predict_bin_on_image(X,rs,cs)]
    def _predict_bin(self, X):
        assert self.is_trained, "Run fit() first"
        return (X[self.ftr_idx,...] < self.thr).astype(np.uint8)
    def _predict_bin_on_image(self, X, rs, cs):
        assert self.is_trained, "Run fit() first"
        r, c, ch = self.ftr
        chs = np.full_like(rs, ch)
        return (X[rs+r,cs+c,chs]<self.thr).astype(np.uint8)
    def as_dict(self):
        return {
            "ftr": list(map(int, self.ftr)),
            "thr": float(self.thr),
            "pred": self.predictions.astype("f").tolist(),
        }
    @property
    def is_trained(self):
        return self.ftr_idx is not None
    def __repr__(self):
        if self.is_trained:
            return f"{self.__class__} shape={self.shape}, ftr_idx={self.ftr_idx} {self.ftr}, thr={self.thr:0.3f}, preds={self.predictions} ({self.predictions.dtype}), err={self.err:0.2f}"
        else:
            return f"{self.__class__} [uninitialized]"


class DTree:
    """
    """
    def __init__(self, shape, depth=2, banks=None):
        self.shape = shape
        self.depth = depth
        self.banks = banks
        self.nodes = None
        self.leaf = None
    def fit(self, X0, W0, X1, W1, quantizer=None):
        if self.is_trained:
            logger.warning("Discarding trained tree")
        n_nodes = (2**self.depth-1)
        self.nodes = [None] * n_nodes
        self.leaf = [None] * n_nodes
        def fit_node(X0, W0, X1, W1, level, position):
            #print(f"fitting node at level {level}, to position {position}")
            node = DStump(self.shape).fit(X0, W0, X1, W1, quantizer)
            self.nodes[position-1] = node
            self.leaf[position-1] = level >= self.depth
            if level < self.depth:
                b0 = node._predict_bin(X0)
                b1 = node._predict_bin(X1)
                logger.debug(f"{(b1==0).sum() + (b0==0).sum()} -> left; {(b1==1).sum() + (b0==1).sum()} -> right")
                fit_node(X0[...,b0==0], W0[b0==0], X1[...,b1==0], W1[b1==0], level+1, position=2*position)
                fit_node(X0[...,b0==1], W0[b0==1], X1[...,b1==1], W1[b1==1], level+1, position=2*position+1)
        fit_node(X0, W0, X1, W1, level=1, position=1)
        return self
    def predict(self, X):
        def predict_node(X, node_id):
            if self.leaf[node_id-1]:
                return self.nodes[node_id-1].predict(X)
            else:
                bin = self.nodes[node_id-1]._predict_bin(X)
                h = np.empty(X.shape[1], "f")
                h[bin==0] = predict_node(X[...,bin==0], 2*node_id)
                h[bin==1] = predict_node(X[...,bin==1], 2*node_id+1)
                return h
        assert self.is_trained, "Run fit() first"
        return predict_node(X, 1)
    def predict_on_image(self, X, rs, cs):
        def predict_node(X, rs, cs, node_id):
            if self.leaf[node_id-1]:
                return self.nodes[node_id-1].predict_on_image(X, rs, cs)
            else:
                bin = self.nodes[node_id-1]._predict_bin_on_image(X, rs, cs)
                h = np.empty(rs.size, np.float32)
                h[bin==0] = predict_node(X, rs[bin==0], cs[bin==0], 2*node_id)
                h[bin==1] = predict_node(X, rs[bin==1], cs[bin==1], 2*node_id+1)
                return h
        assert self.is_trained, "Run fit() first"
        return predict_node(X, rs, cs, 1)
    def as_dict(self):
        d = {"ftr":[], "thr":[], "pred":[]}
        for node,leaf_flag in zip(self.nodes, self.leaf):
            nd = node.as_dict()
            d["ftr"].append(nd["ftr"])
            d["thr"].append(nd["thr"])
            d["pred"].append(nd["pred"] if leaf_flag else None)
        return d
    @property
    def is_trained(self):
        return self.nodes is not None
    # def __repr__(self):
    #     pass


default_channel_opts = {
    "shrink": 2,
    "n_per_oct": 4,
    "smooth": 0,
    "target_dtype": np.int16,
    "channels": []
}


def weights(H):
    return np.exp(H) / H.size / 2


def normalized_weights(W0, W1):
    w = W0.sum() + W1.sum()
    return W0/w, W1/w


def loss(H0, H1):
    return (np.exp(H0).sum() + np.exp(-H1).sum()) / (H0.size+H1.size)


class Model:
    def __init__(self, shape, channel_opts, alpha=0.1, n_pos=2000, n_neg=2000, tr_set_size=0, quantizer=None, logger=None, wh=DStump, **wh_args):
        self.logger = logger = logging.getLogger(__name__)
        self.shape = shape
        self.channel_opts = channel_opts
        self.alpha = alpha
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.tr_set_size = tr_set_size
        self.quantizer = quantizer
        self.wh = wh
        self.wh_args = wh_args
        self.classifier = []
        self.P0 = 1
        self.P1 = 1

    def fit_stage(self, X0, H0, X1, H1, theta=None):
        W0 = weights(H0)
        W1 = weights(-H1)
        if self.tr_set_size > 0:
            idx0 = np.random.choice(W0.size, self.tr_set_size, p=W0/W0.sum())
            idx1 = np.random.choice(W1.size, self.tr_set_size, p=W1/W1.sum())
            W0 = weights( H0[idx0])
            W1 = weights(-H1[idx1])
            weak = self.wh(self.shape, **self.wh_args).fit(X0[...,idx0], W0, X1[...,idx1], W1, self.quantizer)
        else:
            weak = self.wh(self.shape, **self.wh_args).fit(X0, W0, X1, W1, self.quantizer)
        H0 += weak.predict(X0)
        H1 += weak.predict(X1)
        if theta is None:
            theta = fit_rejection_threshold(H0, self.P0, H1, self.P1, self.alpha)
        return weak, theta

    def fit(self, generator, T):
        pool = SamplePool(self.shape, self.channel_opts, n_pos=self.n_pos, n_neg=self.n_neg)
        history = []
        for t in range(T):
            logger.info(f"Training stage {t+1}/{T}")
            pool.update(generator, self.as_dict())
            X1,H1 = pool.get_positive()
            X0,H0 = pool.get_negative()

            logging.debug(f"Loss: {loss(H0,H1):0.5f}")
            F0 = image_to_features(X0)
            F1 = image_to_features(X1)

            Y1,mask = self.predict(F1)
            assert np.all(mask)
            assert np.all(H1 == Y1)

            theta = None if 2 < t <= 32 else -np.inf
            weak, theta = self.fit_stage(F0, H0, F1, H1, theta)

            # p = np.sum(H0 > theta) / H0.size
            # if theta > -np.inf and p > 0.95:
            #     logger.info(f"Neg probability too high {p:.2f} (require < 0.95). Forcing theta to -inf")
            #     theta = -np.inf
            self.classifier.append( (weak, theta) )
            p0,p1 = pool.prune(theta)
            self.P0 *= p0
            self.P1 *= p1
            logging.info(f"Negative rejection {1-self.P0:0.4f}")
            logging.info(f"Positive rejection {1-self.P1:0.4f}")
            history.append( (self.P0, self.P1,loss(H0,H1)) )
        return history

    def predict(self, X):
        n = X.shape[1]
        H = np.zeros(n, np.float32)
        mask = np.ones(n, np.bool)
        for weak, theta in self.classifier:
            H[mask] += weak.predict(X[...,mask])
            if theta == -np.inf:
                continue
            mask = np.logical_and(mask, H>=theta)
        return H, mask

    def as_dict(self):
        model_dict = {
            "shape": self.shape,
            "channel_opts": self.channel_opts.copy(),
            "quantizer": self.quantizer.copy() if self.quantizer is not None else None,
            "classifier": self.classifier.copy()
        }
        return model_dict

    def export(self, filename):
        d = self.as_dict()
        model_dict = {}
        model_dict["shape"] = d["shape"]
        ch = d["channel_opts"]
        ch["target_dtype"] = ch["target_dtype"].__name__
        ch["channels"] = [ (f.__name__,p) for f,p in ch["channels"] ]
        model_dict["channel_opts"] = ch
        model_dict["classifier"] = [ (w.as_dict(),float(t))  for w,t in d["classifier"] ]
        import json
        with open(filename, "w") as f:
            json.dump(model_dict, f)


def fit_rejection_threshold(H0, P0, H1, P1, alpha):
    """
    Fit threshold according to SPRT.
    """
    max0 = np.max(H0)
    min1 = np.min(H1)
    if max0 < min1:
        logger.debug(f"H0 and H1 are non-overlapping H0 < {max0}, H1 > {min1}")
        return min1
    ts = np.concatenate([H0,H1])
    ts = np.sort(np.unique(ts))
    #print(ts, ts.dtype)
    if ts.size < 3:
        logger.debug(f"Not enough unique responses to estimate theta (forcing to {-np.inf})")
        return -np.inf
    ts = ts[1:]
    R = np.empty_like(ts)
    logger.debug(f"Testing {R.size} thresholds on interval <{min(ts):.2f},{max(ts):.2f}>")
    for i,t in enumerate(ts):
        p0 = (H0 < t).sum() / H0.size
        p1 = (H1 < t).sum() / H1.size
        R[i] = (P0 * p0 + (1-P0)) / (P1 * p1 + (1-P1))
    A = 1 / alpha
    logger.debug(f"R: <{min(R):.2f},{max(R):.2f}>; need R > {A}")
    idx = np.nonzero(R > A)[0]
    if idx.size == 0:
        logger.debug(f"No suitable theta found")
        theta = -np.inf
    else:
        theta = ts[np.max(idx)]
    logger.debug(f"theta = {theta:.4f}")
    return theta
