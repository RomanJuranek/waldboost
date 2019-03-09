import logging
import numpy as np
from itertools import combinations


logger = logging.getLogger(__name__)


def find_threshold(f0, w0, f1, w1, edges=None):
    if edges is None:
        f = np.concatenate([f0,f1])
        emin = np.min(f) - 1e-3
        emax = np.max(f) + 1e-3
        edges = np.linspace(emin, emax, 64)
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


def fit_weak_learner(f0, w0, f1, w1, eps=1e-4):
    # Todo use CDF to better estimate thresholds
    # minZ = 1e10
    # for t in (2*np.random.rand(40))-1:
    #     wsum0,_ = np.histogram(f0>t, bins=[0,1,2], weights=w0)
    #     wsum1,_ = np.histogram(f1>t, bins=[0,1,2], weights=w1)
    #     Z = 2 * np.sqrt(wsum0 * wsum1).sum()
    #     if Z < minZ:
    #         minZ = Z
    #         hs = 0.5 * np.log((wsum1+eps) / (wsum0+eps))
    #         thr = t
    edges = np.linspace(-2, 2, 128)
    thr = find_threshold(f0, w0, f1, w1, edges)
    wsum0,_ = np.histogram(f0>thr, bins=[0,1,2], weights=w0)
    wsum1,_ = np.histogram(f1>thr, bins=[0,1,2], weights=w1)
    hs = 0.5 * np.log((wsum1+eps) / (wsum0+eps))
    Z = 2 * np.sqrt(wsum0 * wsum1).sum()
    return Z, thr, hs


def takemin(iterable, objective):
    take_el = None
    best = None
    best_idx = None
    for idx, el in enumerate(iterable):
        m = objective(el)
        if best is None or m < best:
            best = m
            take_el = el
            best_idx = idx
    return best_idx, take_el


def get_best_weak_learner(X0, W0, X1, W1):
    res = map(lambda f0,f1: fit_weak_learner(f0, W0, f1, W1), X0, X1)
    params, (Z, thr, hs) = takemin(res, lambda x: x[0])
    logger.debug(f"Found Z = {Z:0.2f}, params = {params}")
    return params, thr, hs


def fit_stage(
    S0, S1,
    feature_params,
    alpha=0.1,
    theta=None):

    # Unpack data and get sample weights
    X0, H0, P0 = S0
    X1, H1, P1 = S1
    N0 = X0.shape[0]
    N1 = X1.shape[0]
    W0 = np.exp(H0) / N0 / 2
    W1 = np.exp(-H1) / N1 / 2

    # Take samples for training
    logger.debug(f"Selecting training samples")
    i0 = np.random.choice(N0, 1500, p=W0/W0.sum())
    i1 = np.random.choice(N1, 1500, p=W1/W1.sum())

    wts0 = W0[i0]
    wts1 = W1[i1]
    w = wts0.sum() + wts1.sum()
    wts0 /= w
    wts1 /= w

    selector = np.random.choice(len(feature_params), 1000)
    selected_params = [feature_params[s] for s in selector]
    F0 = [pixel_comparison(X0[i0], p) for p in selected_params]
    F1 = [pixel_comparison(X1[i1], p) for p in selected_params]
    params, thr, hs = get_best_weak_learner(F0, wts0, F1, wts1)
    params = selected_params[params]

    logger.debug("Updating training set")
    f0 = pixel_comparison(X0, params)
    f1 = pixel_comparison(X1, params)
    H0 += hs[(f0 > thr)*1]
    H1 += hs[(f1 > thr)*1]

    logger.debug("Training rejection threshold")
    if theta is None:
        theta = waldboost_threshold(H0, P0, H1, P1, alpha)

    return params, thr, hs, theta


def waldboost_threshold(H0, P0, H1, P1, alpha):
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


def pixel_comparison(X, params):
    r0, c0, r1, c1 = params
    return X[:,r0,c0] - X[:,r1,c1]


def feature_params(shape, max_dist = 3):
    u,v = shape
    r,c = np.mgrid[:u,:v]
    r = r.ravel()
    c = c.ravel()
    params = []
    for p0, p1 in combinations(zip(r,c), 2):
        d = np.linalg.norm(np.array(p1) - np.array(p0))
        if 0 < d < max_dist:
            params.append(p0 + p1)
    logger.debug(f"{len(params)} features generated")
    return params
