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


logger = logging.getLogger(__name__)


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


def fit_decision_stump(f0, w0, f1, w1, eps=1e-4):
    thr = find_threshold(f0, w0, f1, w1)
    wsum0,_ = np.histogram((f0<thr).astype(np.uint8), bins=[0,1,2], weights=w0)
    wsum1,_ = np.histogram((f1<thr).astype(np.uint8), bins=[0,1,2], weights=w1)
    hs = 0.5 * np.log((wsum1+eps) / (wsum0+eps))
    Z = 2 * np.sqrt(wsum0 * wsum1).sum()
    return (thr, hs), Z


def normalized_weights(W0, W1):
    w = W0.sum() + W1.sum()
    return W0/w, W1/w


class DecisionStump:
    """
    Simple, threshold-based classifier.
    It optimizes thresholds for all features, picks the one
    with minimal error and trains response table (2 items).
    It is represented by feature id, threshold value and the response table.

    DecisionStump.fit fits the classifier and returns
    new instance. eval returns responses for given samples.

    Example:
    c = DecisionStump.fit(X0, W0, X1, W1)
    Y0 = c.eval(X0)
    Y1 = c.eval(X1)
    """
    def __init__(self, ftr_idx, thr, hs, shape=None):
        self.ftr_idx = ftr_idx
        self.thr = thr
        self.hs = hs
        self.ftr = None
        if shape is not None:
            self.ftr = np.unravel_index(ftr_idx, shape)
        logger.debug(f"DStump {ftr_idx}, {thr}, {hs.tolist()}, {self.ftr}")

    @classmethod
    def fit(cls, X0, W0, X1, W1, shape):
        min_err = None
        weak = None
        w0,w1 = normalized_weights(W0, W1)
        for ftr,(x0,x1) in enumerate(zip(X0,X1)):
            (thr, hs), err = fit_decision_stump(x0,w0,x1,w1)
            if min_err is None or err < min_err:
                weak = (ftr, thr, hs)
                min_err = err
        ftr, thr, hs = weak
        return cls(ftr, thr, hs, shape)

    def eval(self, X):
        bin = self.eval_bin(X)
        return self.hs[bin]

    def eval_bin(self, X):
        return (X[self.ftr_idx,...] < self.thr).astype(np.uint8)

    def eval_bin_on_image(self, X, rs, cs):
        r0, c0, ch0 = self.ftr
        chs = np.full_like(rs, ch0)
        return (X[rs+r0,cs+c0,chs] < self.thr).astype(np.uint8)

    def eval_on_image(self, X, rs, cs):
        bin = self.eval_bin_on_image(X, rs, cs)
        return self.hs[bin]

    def as_dict(self):
        return {"ftr": list(map(int,self.ftr)), "thr": int(self.thr), "predictions": list(map(float,self.hs.tolist())) }


class DecisionTree:
    """
    Decision tree weak classifier.
    Each node is DecisionStump optimized on a subset of data enering the node.

    DecisionTree.fit(...) fits the classifier and returns initialized instance
    with eval() method.

    Example:
    c = DecisionTree.fit(X0, W0, X1, W1)
    Y0 = c.eval(X0)
    Y1 = c.eval(X1)
    """
    def __init__(self, root):
        self.root = root

    @staticmethod
    def fit_node(X0, W0, X1, W1, depth, shape):
        node = DecisionStump.fit(X0, W0, X1, W1, shape)
        b0 = node.eval_bin(X0)
        b1 = node.eval_bin(X1)
        logger.debug(f"{(b1==0).sum() + (b0==0).sum()} -> left; {(b1==1).sum() + (b0==1).sum()} -> right")
        if depth > 1:
            left =  DecisionTree.fit_node(X0[...,b0==0], W0[b0==0], X1[...,b1==0], W1[b1==0], depth-1, shape)
            right = DecisionTree.fit_node(X0[...,b0==1], W0[b0==1], X1[...,b1==1], W1[b1==1], depth-1, shape)
        else:
            left = right = None
        return node, left, right

    @classmethod
    def fit(cls, X0, W0, X1, W1, depth=2, shape=None):
        root_node = cls.fit_node(X0, W0, X1, W1, depth, shape)
        return cls(root_node)

    def eval(self, X):
        def eval_node(X, node):
            root, left, right = node
            bin = root.eval_bin(X)
            if left is None:
                assert right is None, "Inconsistent tree"
                return root.hs[bin]
            h0 = eval_node(X[...,bin==0], left)
            h1 = eval_node(X[...,bin==1], right)
            h = np.empty(X.shape[1], "f")
            h[bin==0] = h0
            h[bin==1] = h1
            return h
        return eval_node(X, self.root)

    def eval_on_image(self, X, rs, cs):
        def eval_node(X, rs, cs, node):
            root, left, right = node
            bin = root.eval_bin_on_image(X, rs, cs)
            if left is None:
                assert right is None
                return root.hs[bin]
            response = np.empty(rs.size, np.float32)
            response[bin==0] = eval_node(X, rs[bin==0], cs[bin==0], left)
            response[bin==1] = eval_node(X, rs[bin==1], cs[bin==1], right)
            return response
        return eval_node(X, rs, cs, self.root)

    def as_dict(self):
        def node_dict(node):
            root, left, right = node
            d = [ root.as_dict() ]
            if left is not None:
                d.append(node_dict(left))
                d.append(node_dict(right))
            return d
        return node_dict(self.root)


def fit_stage(
    X0, H0, P0, X1, H1, P1,
    wh=DecisionStump,
    alpha=0.1, theta=None,
    shape=None):
    """
    Warning:
    The function mutates values in H0 and H1.
    """

    N0 = X0.shape[0]
    N1 = X1.shape[0]
    W0 = np.exp( H0) / N0 / 2
    W1 = np.exp(-H1) / N1 / 2

    weak = wh.fit(X0, W0, X1, W1, shape=shape)
    h0 = weak.eval(X0)
    h1 = weak.eval(X1)
    logger.debug(f"Update H0 {H0.shape} with {h0.shape}")
    H0 += h0
    logger.debug(f"Update H1 {H1.shape} with {h1.shape}")
    H1 += h1

    if theta is None:
        theta = fit_rejection_threshold(H0, P0, H1, P1, alpha)

    return weak, theta


def fit_rejection_threshold(H0, P0, H1, P1, alpha):
    """
    Fit threshold according to SPRT.
    """
    max0 = np.max(H0)
    min1 = np.min(H1)
    if max0 < min1:
        logger.debug(f"H0 and H1 are non-overlapping H0 < {max0}, H1 > {min1}")
        return 0.5 * (max0 + min1)
    H = np.concatenate([H0,H1])
    H = np.sort(np.unique(H))
    ts = 0.5 * (H[1:] + H[:-1])
    R = np.empty_like(ts)
    logger.debug(f"Testing {R.size} thresholds on interval <{min(H):.2f},{max(H):.2f}>")
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
    logger.debug(f"theta = {theta:.2f}")
    return theta


def fit_model(model, pool, alpha=0.1, T=1024, wh=DecisionStump):
    """
    """
    shape = model["opts"]["shape"]
    model["wh_class"] = wh
    for t in range(T):
        logger.info(f"Training stage {t+1}/{T}")

        pool.update(model)
        X1,H1,P1 = pool.get_positive()
        X0,H0,P0 = pool.get_negative()

        F0 = np.moveaxis(X0,0,-1).reshape(-1,H0.size) # Transform (N,H,W) -> (HxW,N)
        F1 = np.moveaxis(X1,0,-1).reshape(-1,H1.size) # Transform (N,H,W) -> (HxW,N)

        theta = None if t%8==7 and t<64 else -np.inf
        weak, theta = fit_stage(F0, H0, P0, F1, H1, P1, wh=wh, alpha=alpha, theta=theta, shape=shape)

        # Check negative probability induced by theta
        p = np.sum(H0 > theta) / H0.size
        if theta > -np.inf and p > 0.95:
            logger.info(f"Neg probability too high {p:.2f} (require < 0.95). Forcing theta to -inf")
            theta = -np.inf

        # Add new stage to the model
        model["classifier"].append( (weak, theta) )

        # Remove samples from the pool and update probabilites
        pool.prune(theta)


def save_model(model, filename):
    cl = []
    for weak,theta in model["classifier"]:
        cl.append( [weak.as_dict(), float(theta)] )
    import json
    with open(filename,"w") as f:
        json.dump(cl, f, indent=True)
