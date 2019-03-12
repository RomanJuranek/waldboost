""" WaldBoost training algortithm

Support for training of waldboost classifiers. The functions fit_stage accepts
training data (negative and positive samples X, their responses H and priors P)
and returns new weak classifier of specified type. The module implements
a DecisionStump class for simple threshold-based weak classifiers.

See also:
waldboost.training.DecisionStump
waldboost.training.fit_stage
"""


import logging
import numpy as np


logger = logging.getLogger(__name__)


def find_threshold(f0, w0, f1, w1, edges=None):
    """
    Find threshold to separate two distributions with minimal error.
    """
    if edges is None:
        f = np.concatenate([f0,f1])
        emin = np.min(f) - 1e-3
        emax = np.max(f) + 1e-3
        edges = np.linspace(emin, emax, 256)
        #print(emin, emax, edges)
    else:
        f0 = np.clip(f0, edges[0], edges[-1])
        f1 = np.clip(f1, edges[0], edges[-1])
    w0 = w0 / w0.sum()
    w1 = w1 / w1.sum()
    p0,_ = np.histogram(f0, bins=edges, weights=w0)
    p1,_ = np.histogram(f1, bins=edges, weights=w1)
    cdf0 = np.cumsum(p0)
    cdf1 = np.cumsum(p1)
    #print(cdf0, cdf1)
    err = 2 * np.sqrt(cdf0*cdf1 + (1-cdf0)*(1-cdf1))
    k = np.argmin(err)
    return edges[k+1]


def fit_decision_stump(f0, w0, f1, w1, eps=1e-4, edges=None):
    thr = find_threshold(f0, w0, f1, w1, edges)
    wsum0,_ = np.histogram((f0>thr).astype(np.uint8), bins=[0,1,2], weights=w0)
    wsum1,_ = np.histogram((f1>thr).astype(np.uint8), bins=[0,1,2], weights=w1)
    hs = 0.5 * np.log((wsum1+eps) / (wsum0+eps))
    Z = 2 * np.sqrt(wsum0 * wsum1).sum()
    return (thr, hs), Z


def normalized_weights(W0, W1):
    w = W0.sum() + W1.sum()
    return W0/w, W1/w


class DecisionStump:
    def __init__(self, ftr, thr, hs):
        self.ftr = ftr
        self.thr = thr
        self.hs = hs

    @classmethod
    def fit(cls, X0, W0, X1, W1):
        min_err = None
        weak = None
        w0,w1 = normalized_weights(W0, W1)
        for ftr,(x0,x1) in enumerate(zip(X0,X1)):
            (thr, hs), err = fit_decision_stump(x0,w0,x1,w1)
            if min_err is None or err < min_err:
                weak = (ftr, thr, hs)
                min_err = err
        ftr, thr, hs = weak
        return cls(ftr, thr, hs)

    def eval(self, X):
        bin = (X[self.ftr,...] > self.thr).astype(np.uint8)
        return self.hs[bin]

    def as_tuple(self):
        return self.ftr, self.thr, self.hs


def fit_stage(
    X0, H0, P0, X1, H1, P1,
    wh = DecisionStump,
    alpha=0.1, theta=None):
    """
    Some sensible doc here
    """

    N0 = X0.shape[0]
    N1 = X1.shape[0]
    W0 = np.exp( H0) / N0 / 2
    W1 = np.exp(-H1) / N1 / 2

    weak = wh.fit(X0, W0, X1, W1)
    h0 = weak.eval(X0)
    h1 = weak.eval(X1)
    H0 += h0
    H1 += h1

    if theta is None:
        theta = fit_rejection_threshold(H0, P0, H1, P1, alpha)

    return weak, theta


def fit_rejection_threshold(H0, P0, H1, P1, alpha):
    """
    Fit threshold accorting to SPRT.
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
