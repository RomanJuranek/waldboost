""" WaldBoost training algortithm """


import logging
import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier


logger = logging.getLogger(__name__)


def weights(H):
    return np.exp(H) / H.size / 2


def as_features(X):
    n,*shape = X.shape
    return X.reshape( (n,np.prod(shape)) )


class DTree:
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
        return DTree(feature, tree.threshold, tree.children_left, tree.children_right, pred)
    @staticmethod
    def from_proto(proto):
        ftr = np.array(proto.feature).reshape((-1,3))
        ftr = [ tuple(x) if x[0]>=0 else None for x in ftr ]
        thr = np.array(proto.threshold)
        left = np.array(proto.left)
        right = np.array(proto.right)
        pred = np.array(proto.prediction)
        return DTree(ftr, thr, left, right, pred)
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
    def apply(self, X):
        node = np.zeros(X.shape[0], "i")
        for n,(ftr,t,lnode,rnode) in enumerate(zip(self.feature, self.threshold, self.left, self.right)):
            if ftr is None: continue
            r,c,ch = ftr
            idx = np.flatnonzero(node == n)
            bin = X[idx,r,c,ch] <= t
            node[idx[ bin]] = lnode
            node[idx[~bin]] = rnode
        return node
    def predict(self, X):
        return self.prediction[self.apply(X)]
    def predict_on_image(self, X, rs, cs):
        node = np.zeros(rs.size,"i")
        for n,(ftr,t,lnode,rnode) in enumerate(zip(self.feature, self.threshold, self.left, self.right)):
            if ftr is None: continue
            r,c,ch = ftr
            idx = np.flatnonzero(node == n)
            bin = X[rs[idx]+r,cs[idx]+c,ch] <= t
            node[idx[ bin]] = lnode
            node[idx[~bin]] = rnode
        return self.prediction[node]


# def sample_training_data(X, H, W, n_samples=None, trim_ratio=0.2):
#     if n_samples and n_samples < H.size:
#         idx = weighted_sampling(n_samples, p=W/W.sum(), trim_ratio=trim_ratio)
#         _X,_H = X[idx,...], H[idx]
#     else:
#         _X,_H = X, H
#     return _X, _H


def loss(H0, H1):
    W0 = weights(H0)
    W1 = weights(-H1)
    return W0.mean() + W1.mean()


class Learner:
    """ Training algorithm """
    def __init__(self, alpha=0.1, wh=DTree, **wh_args):
        self.alpha = alpha
        self.wh = wh
        self.wh_args = wh_args
        self.p0 = []
        self.p1 = []
        self.losses = []

    @staticmethod
    def from_dict(d):
        L = Learner(alpha=d["alpha"], wh=d["wh"], **d["wh_args"])
        L.p0 = d["p0"]
        L.p1 = d["p1"]
        L.losses = d["losses"]
        if len(L.p0) != len(L.losses) or len(L.p1) != len(L.losses):
            raise ValueError("Wrong values for p0, p1 or loss")
        return L

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return Learner.from_dict(pickle.load(f))

    @property
    def false_positive_rate(self):
        return np.prod(self.p0)

    @property
    def true_positive_rate(self):
        return np.prod(self.p1)

    @property
    def loss(self):
        return self.losses[-1] if self.losses else None

    def __len__(self):
        return len(self.losses)

    def __bool__(self):
        return True

    def get_stats(self):
        return {
            "false_positive_rate": np.cumprod(self.p0),
            "true_positive_rate": np.cumprod(self.p1),
            "loss": np.array(self.losses),
        }

    def fit_stage(self, model, X0, H0, X1, H1, theta=None, **wh_args):
        """ Append new stage to the model """
        W0 = weights(H0)
        W1 = weights(-H1)

        weak = self.wh.fit(X0, W0, X1, W1, **{**self.wh_args, **wh_args})

        # Update H
        H0 = H0 + weak.predict(X0)
        H1 = H1 + weak.predict(X1)

        # Fit threshold
        if not theta:
            theta = fit_rejection_threshold(H0, self.false_positive_rate, H1, self.true_positive_rate, self.alpha)

        # calc p and update P
        p0 = (H0>=theta).sum() / H0.size
        p1 = (H1>=theta).sum() / H1.size
        self.p0.append(p0)
        self.p1.append(p1)
        self.losses.append(loss(H0, H1))

        model.append(weak, theta)

        return self.loss, self.false_positive_rate, self.true_positive_rate


def fit_rejection_threshold(H0, P0, H1, P1, alpha):
    """ Fit threshold according to SPRT. """
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


class BasicRejectionSchedule:
    def __init__(self, rejection_interval=(0,None), target_p0=1e-5):
        """
        Inputs
        ------
        rejection_interval : tuple or None
            Interval on which theta should be learned. Outside of this interval, the
            value is forced to -np.inf. Tuple (s0,s1) means first and last stage.
            When s0 or s1 is None, s0 defaults to 0, s1 to np.inf. When None is given,
            s0=0, s1=np.inf.
        target_p0 : scalar float
            Condition for stopping rejection of training samples. If 0 < stop_sampling < 1
            (usually close to zero, like 1e-5), rejection stops when p0 (false positive
            probability) reaches below this threshold.

        Output
        ------
        theta : float or None
            The value of theta, either -np.inf or None. None value means that theta must be
            estimated from data.
        """
        if rejection_interval is None:
            rejection_interval = (None, None)
        self.s0 = rejection_interval[0] or 0
        self.s1 = rejection_interval[1] or np.inf
        self.target_p0 = target_p0

    def __call__(self, stage, p0):
        if stage < self.s0 or stage > self.s1 or p0 < self.target_p0:
            return -np.inf
        return None
