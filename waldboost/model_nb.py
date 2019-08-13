import numpy as np
from numba import jit, jitclass, int32, float32


dtree_spec = [
    ('feature', int32[:,:,:]),    # (T,N,3)
    ('threshold', float32[:,:]),  # (T,N)
    ('left', int32[:,:]),         # (T,N)
    ('right', int32[:,:]),        # (T,N)
    ('prediction', float32[:,:]), # (T,N)
    ('theta', float32[:]),        # (T)
]


@jitclass(dtree_spec)
class Model(object):
    def __init__(self, feature, threshold, left, right, prediction, theta):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.theta = theta

    def channels(self, X):
        for

    def predict_stage_on_image(self, X, rs, cs):
        node = np.zeros(rs.size, int32)
        for k in range(rs.size):
            n = int32(0)
            while (self.left[n] != -1):
                _r,_c,_ch = self.feature[n,:]
                split = X[rs[k]+_r,cs[k]+_c,_ch] <= self.threshold[n]
                n = self.left[n] if split else self.right[n]
            node[k] = n
        return self.prediction[node]

    def predict_on_image(self, X):
        u,v,ch_image = X.shape
        m,n,ch_cls = self.shape
        idx = np.arange(max(u-m,0)*max(v-n,0), dtype=np.int32)
        rs = idx % (u-m)
        cs = idx // (u-m)
        hs = np.zeros_like(rs, np.float32)
        for T in range(self.size):
            if rs.size == 0: break
            hs += self.predict_stage_on_image(X, rs, cs)
            if theta == -np.inf:
                continue
            mask = hs >= theta
            rs = rs[mask]
            cs = cs[mask]
            hs = hs[mask]
        return rs, cs, hs

    @property
    def size(self):
        return self.feature.shape[0]


def transform_model(model):
    T = len(model)
    N = max(len(w.feature) for w,_ in model.classifier)

    ftr = np.zeros((T,N,3), "i")
    thr = np.zeros((T,N), "f")
    left = np.zeros((T,N), "i")
    right = np.zeros((T,N), "i")
    pred = np.zeros((T,N), "f")
    th = np.zeros(T, "f")

    for t,(weak,theta) in enumerate(model.classifier):
        n = len(weak.feature)
        ftr[t,:n,...] = np.array([f if f is not None else (-1,-1,-1) for f in weak.feature],"i")
        thr[t,:n] = weak.threshold.astype("f")
        left[t,:n] = weak.left.astype("i")
        right[t,:n] = weak.right.astype("i")
        pred[t,:n] = weak.prediction.astype("f")
        th[t] = theta

    return ftr, thr, left, right, pred, th
