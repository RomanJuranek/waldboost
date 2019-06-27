""" WaldBoost training algortithm
"""


import logging
import numpy as np


logger = logging.getLogger(__name__)


def iter_features(X):
    """
    Sequence of X[:,a,b,c,...]
    """
    n,*shape = X.shape
    _X = X.reshape(n,-1)
    for k in range(_X.shape[-1]):
        yield _X[...,k]


def weights(H):
    return np.exp(H) / H.size / 2


def normalized_weights(W0, W1):
    w = W0.sum() + W1.sum()
    return W0/w, W1/w


def loss(H0, H1):
    w0, w1 = normalized_weights(weights(H0), weights(-H1))
    return (w0.sum() + w1.sum()) / (H0.size+H1.size)


def weighted_sampling(n, p, trim_ratio=0.5):
    assert 0 <= trim_ratio <= 1, "Trim ratio must be in <0,1> interval"
    order = np.argsort(p)[::-1]
    n_trim = round(n * trim_ratio)
    trimmed, rest = np.split(order, [n_trim])
    p_rest = p[rest] / p[rest].sum()
    selected = np.random.choice(rest, n-n_trim, p=p_rest, replace=False)
    return np.concatenate([trimmed, selected])


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
    p0,_ = np.histogram(f0, bins=edges, weights=w0)
    p1,_ = np.histogram(f1, bins=edges, weights=w1)
    cdf0 = np.cumsum(p0)
    cdf1 = np.cumsum(p1)
    err = 2 * np.sqrt(cdf0*cdf1 + (1-cdf0)*(1-cdf1))
    k = np.argmin(err)
    return edges[k+1]


def fit_decision_stump(f0, w0, f1, w1, eps=1e-4, min_split_ratio=0.05):
    w0 = w0 / w0.sum()
    w1 = w1 / w1.sum()
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
    def __init__(self, ftr=None, thr=None, h=None, err=None):
        self.ftr = ftr
        self.thr = thr
        self.h = h
        self.err = err
    @staticmethod
    def fit(X0, W0, X1, W1):
        assert X0.shape[1:] == X1.shape[1:]
        shape = X0.shape[1:]
        instance = None
        err = None
        for k,(x0,x1) in enumerate(zip(iter_features(X0),iter_features(X1))):
            (thr, hs), err_k = fit_decision_stump(x0,W0,x1,W1)
            if err is None or err_k < err:
                ftr = np.unravel_index(k, shape)
                err = err_k
                instance = DStump(ftr, thr, hs, err)
        return instance
    def quantize(self, func):
        self.h = func(self.h)
    @staticmethod
    def from_tuple(t):
        return DStump(*t)
    def as_tuple(self):
        return self.ftr, self.thr, self.h, self.err
    @staticmethod
    def from_proto(proto):
        return DStump(tuple(proto.ftr), proto.thr, np.array(proto.pred), proto.err)
    def as_proto(self, proto):
        proto.ClearField("ftr")
        proto.ClearField("pred")
        proto.ftr.extend(self.ftr)
        proto.thr = self.thr
        proto.pred.extend(self.h)
        proto.err = self.err
    def predict(self, X):
        self.check_trained()
        return self.h[self.predict_bin(X)]
    def predict_bin(self, X):
        self.check_trained()
        r, c, ch = self.ftr
        return (X[:,r,c,ch]<self.thr).astype(np.uint8)
    def predict_bin_on_image(self, X, rs, cs):
        self.check_trained()
        r, c, ch = self.ftr
        chs = np.full_like(rs, ch)
        return (X[rs+r,cs+c,chs]<self.thr).astype(np.uint8)
    def predict_on_image(self, X, rs, cs):
        return self.h[self.predict_bin_on_image(X, rs, cs)]
    def check_trained(self):
        assert self.ftr is not None, "Run fit() first"


class DTree:
    def __init__(self, nodes=None, leaf_tag=None):
        self.nodes = nodes
        self.leaf_tag = leaf_tag
    @staticmethod
    def fit(X0, W0, X1, W1, depth=2):
        n_nodes = (2**depth-1)
        nodes = [None] * n_nodes
        leaf_tag = [None] * n_nodes
        def fit_node(X0, W0, X1, W1, level, position):
            node = DStump.fit(X0, W0, X1, W1)
            nodes[position-1] = node
            leaf_tag[position-1] = level >= depth
            if level < depth:
                b0 = node.predict_bin(X0)
                b1 = node.predict_bin(X1)
                fit_node(X0[b0==0], W0[b0==0], X1[b1==0], W1[b1==0], level+1, position=2*position+0)
                fit_node(X0[b0==1], W0[b0==1], X1[b1==1], W1[b1==1], level+1, position=2*position+1)
        fit_node(X0, W0, X1, W1, level=1, position=1)
        return DTree(nodes, leaf_tag)
    def quantize(self, func):
        for n in self.nodes:
            n.quantize(func)
    @staticmethod
    def from_tuple(t):
        nodes, leaf_tag = t
        nodes = [DStump.from_tuple(n) for n in nodes]
        return DTree(nodes, leaf_tag)
    def as_tuple(self):
        return [n.as_tuple() for n in self.nodes], self.leaf_tag
    @staticmethod
    def from_proto(proto):
        nodes = [ DStump.from_proto(p) for p in proto.nodes ]
        leaf_tag = np.array(proto.leaf_tag)
        return DTree(nodes, leaf_tag)
    def as_proto(self, proto):
        proto.ClearField("nodes")
        proto.ClearField("leaf_tag")
        proto.leaf_tag.extend(self.leaf_tag)
        for n in self.nodes:
            n_pb = proto.nodes.add()
            n.as_proto(n_pb)
    def predict(self, X):
        def predict_node(X, node_id):
            if self.leaf_tag[node_id-1]:
                return self.nodes[node_id-1].predict(X)
            else:
                b = self.nodes[node_id-1].predict_bin(X)
                h = np.empty(X.shape[0], "f")
                h[b==0] = predict_node(X[b==0], 2*node_id+0)
                h[b==1] = predict_node(X[b==1], 2*node_id+1)
                return h
        self.check_trained()
        return predict_node(X, 1)
    def predict_on_image(self, X, rs, cs):
        def predict_node(rs, cs, node_id):
            if self.leaf_tag[node_id-1]:
                return self.nodes[node_id-1].predict_on_image(X, rs, cs)
            else:
                b = self.nodes[node_id-1].predict_bin_on_image(X, rs, cs)
                h = np.empty(rs.size, np.float32)
                h[b==0] = predict_node(rs[b==0], cs[b==0], 2*node_id+0)
                h[b==1] = predict_node(rs[b==1], cs[b==1], 2*node_id+1)
                return h
        self.check_trained()
        return predict_node(rs, cs, 1)
    def check_trained(self):
        assert all(self.nodes), "Run fit() first"


def as_features(X):
    n,*shape = X.shape
    return X.reshape( (n,np.prod(shape)) )


from sklearn.tree import DecisionTreeClassifier


class SKLearnDTree:
    def __init__(self, feature, threshold, left, right, prediction):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
    @staticmethod
    def fit(X0, W0, X1, W1, **kwargs):
        X = np.concatenate( [as_features(X0), as_features(X1)] )
        Y = np.array([0]*W0.size + [1]*W1.size)
        W = np.concatenate( [W0, W1] )
        T = DecisionTreeClassifier(class_weight="balanced", **kwargs)
        T.fit(X, Y, sample_weight=W)
        _,*shape = X0.shape
        tree = T.tree_
        feature = [ np.unravel_index(f, shape) if f >= 0 else None for f in tree.feature ]
        leaf = T.apply(X)
        pred = np.empty(tree.node_count)
        for n in range(tree.node_count):
            mask = leaf == n
            w0 = (W*mask*(Y==0)).sum() + 1e-3
            w1 = (W*mask*(Y==1)).sum() + 1e-3
            pred[n] = np.log(w1 / w0) / 2
        return SKLearnDTree(feature, tree.threshold, tree.children_left, tree.children_right, pred)
    @staticmethod
    def from_proto(proto):
        ftr = np.array(proto.feature).reshape((-1,3))
        ftr = [ tuple(x) if x[0]>=0 else None for x in ftr ]
        thr = np.array(proto.threshold)
        left = np.array(proto.left)
        right = np.array(proto.right)
        pred = np.array(proto.prediction)
        return SKLearnDTree(ftr, thr, left, right, pred)
    def as_proto(self, proto):
        proto.Clear()
        proto_ftr = []
        for f in self.feature:
            if f is not None:
                proto_ftr.extend(f)
            else:
                proto_ftr.extend( (-1,-1,-1) )
        proto.feature.extend(proto_ftr)
        proto.threshold.extend(self.threshold)
        proto.left.extend(self.left)
        proto.right.extend(self.right)
        proto.prediction.extend(self.prediction)
    def predict(self, X):
        n = X.shape[0]
        node = np.zeros(n, "i")
        for n,(ftr,t,lnode,rnode) in enumerate(zip(self.feature, self.threshold, self.left, self.right)):
            if ftr is None: continue
            r,c,ch = ftr
            idx = np.flatnonzero(node == n)
            bin = X[idx,r,c,ch] <= t
            node[idx[ bin]] = lnode
            node[idx[~bin]] = rnode
        return self.prediction[node]
    def predict_on_image(self, X, rs, cs):
        n = rs.size
        node = np.zeros(n,"i")
        for n,(ftr,t,lnode,rnode) in enumerate(zip(self.feature, self.threshold, self.left, self.right)):
            if ftr is None: continue
            r,c,ch = ftr
            idx = np.flatnonzero(node == n)
            bin = X[rs[idx]+r,cs[idx]+c,ch] <= t
            node[idx[ bin]] = lnode
            node[idx[~bin]] = rnode
        return self.prediction[node]


def sample_training_data(X, H, W, n_samples=None, trim_ratio=0.2):
    if n_samples and n_samples < H.size:
        idx = weighted_sampling(n_samples, p=W/W.sum(), trim_ratio=trim_ratio)
        _X,_H = X[idx,...], H[idx]
    else:
        _X,_H = X, H
    return _X, _H


class Quantizer:
    def __init__(self, xmax=1, bits=8):
        self.xmax = xmax
        self.multiplier = 2**bits / (2*xmax)
    def __call__(self, x, inverse=False, clip=False):
        if not inverse:
            x = np.clip(x, -self.xmax, self.xmax)
            return np.round(x * self.multiplier)
        else:
            return x / self.multiplier


def fit_weak(X0, H0, X1, H1, wh=DStump, **wh_args):
    W0 = weights(H0)
    W1 = weights(-H1)
    #W0, W1 = normalized_weights(W0, W1)
    weak = wh.fit(X0, W0, X1, W1, **wh_args)
    return weak


class Learner:
    def __init__(self, alpha=0.1, n_samples=None, logger=None, wh=DStump, **wh_args):
        self.logger = logger = logging.getLogger(__name__)
        self.alpha = alpha
        self.n_samples = n_samples
        self.wh = wh
        self.wh_args = wh_args
        self.P0 = 1
        self.P1 = 1

    def fit_stage(self, model, X0, H0, X1, H1, theta=None):
        # w0,w1 = weights(H0),weights(-H1)
        # _X0,_H0 = sample_training_data(X0, H0, w0, self.n_samples)
        # _X1,_H1 = sample_training_data(X1, H1, w1, self.n_samples)
        # print(_X0.shape, _H0.shape)
        # print(_X1.shape, _H1.shape)
        weak = fit_weak(X0, H0, X1, H1, self.wh, **self.wh_args)

        # Update H
        H0 = H0 + weak.predict(X0)
        H1 = H1 + weak.predict(X1)
        # Fit threshold
        if not theta:
            theta = fit_rejection_threshold(H0, self.P0, H1, self.P1, self.alpha)
        # calc p and update P
        p0 = (H0>=theta).sum() / H0.size
        p1 = (H1>=theta).sum() / H1.size
        self.P0 *= p0
        self.P1 *= p1

        model.append(weak, theta)


def fit_rejection_threshold(H0, P0, H1, P1, alpha):
    """
    Fit threshold according to SPRT.
    """
    max0 = np.max(H0)
    min1 = np.min(H1)
    if max0 < min1:
        logger.debug(f"H0 and H1 are non-overlapping H0 < {max0}, H1 > {min1}")
        return min1
    ts = np.concatenate([H0.flatten(),H1.flatten()])
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
