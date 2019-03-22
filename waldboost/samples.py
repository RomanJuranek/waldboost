import logging
import numpy as np
import cv2

from . import groundtruth
from .detector import forward, bbs_from_dets
from .channels import channel_pyramid


def sample_random(chns, shape, sample_prob=1e-3):
    u,v,_ = chns.shape
    m,n,_ = shape
    if u > m and v > n:
        map = np.random.rand(u-m,v-n) < sample_prob
        r,c = np.nonzero(map)
    else:
        r,c = [], []
    return r, c, np.zeros_like(r, np.float32)


def predict_and_sample(chns, detector):
    if detector["classifier"]:
        dets = forward(chns, detector)
    else:
        shape = detector["opts"]["shape"]
        dets = sample_random(chns, shape)
    return dets


def gather_samples(chns, rs, cs, shape):
    #u,v = chns.shape
    m,n,_ = shape
    if rs.size == 0:
        return np.empty((0,)+shape, dtype=chns.dtype)
    X = [ chns[r:r+m,c:c+n,...] for r,c in zip(rs, cs) ]
    return np.array(X)


def sample_from_bbs(chns, shape, bbs):
    u,v,_ = chns.shape
    m,n,_ = shape
    map = np.zeros((max(u-m,0),max(v-n,0)), np.bool)
    for bb in bbs:
        x,y,w,h = bb.astype(np.int)
        map[y-m:y+h,x-n:x+w] = 1
    r,c = np.nonzero(map)
    return r, c, np.zeros_like(r, np.float32)


def reject_samples(H, X, theta):
    mask = H >= theta
    p = mask.sum() / mask.size
    return H[mask], X[mask,...], p


class SamplePool:
    def __init__(self, image_generator, shape, n_neg=1000, n_pos=1000, logger=None):
        self.generator = image_generator
        self.shape = shape
        self.min_neg = n_neg
        self.min_pos = n_pos
        self.logger = logger or logging.getLogger(__name__)
        self.X0 = None
        self.X1 = None

    def update(self, detector):
        req_neg = self.min_neg - self.n_neg
        req_pos = self.min_pos - self.n_pos

        if detector["classifier"]:
            req_pos = 0

        if req_neg <= 0 and req_pos <= 0:
            self.logger.debug("Nothing to update (pool is full)")
            return

        new_X0 = []
        new_H0 = []
        new_X1 = []
        new_H1 = []

        while True:
            self.logger.debug(f"Require {req_neg} negative samples and {req_pos} positive samples")
            im, gt = next(self.generator)
            I = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            for chns, scale in channel_pyramid(im, detector["opts"]):
                if req_neg > 0:
                    r,c,h = predict_and_sample(chns, detector)
                    dt = bbs_from_dets(r, c, self.shape, scale)
                    dt_dist, dt_ign, _ = groundtruth.match(dt, gt)
                    fp = dt_dist > 20
                    dt_ign = np.logical_and(dt_ign, ~   fp)
                    if np.any(fp):
                        fp_idx = np.nonzero(fp)[0]
                        if fp_idx.size > 500:
                            fp_idx = np.random.choice(fp_idx, 500)
                        new_X0.append(gather_samples(chns, r[fp_idx], c[fp_idx], self.shape))
                        new_H0.append(h[fp_idx])
                        req_neg -= fp_idx.size

                    for bb, ign, is_fp in zip(dt, dt_ign, fp):
                        x,y,w,h = bb.astype(int)
                        if ign:
                            c = (0,0,0)
                        elif is_fp:
                            c = (0,255,0)
                        else:
                            c = (0,0,255)
                        cv2.rectangle(I, (x,y),(x+w,y+h), c, 1)

                if req_pos > 0:
                    r,c,h = sample_from_bbs(chns, self.shape, gt[:,:4]*scale)
                    dt = bbs_from_dets(r, c, self.shape, scale)
                    dt_dist, dt_ign, _ = groundtruth.match(dt, gt)
                    tp = np.logical_and(dt_dist<5, ~dt_ign)
                    if np.any(tp):
                        tp = np.nonzero(tp)[0]
                        new_X1.append(gather_samples(chns, r[tp], c[tp], self.shape))
                        new_H1.append(h[tp])
                        req_pos -= tp.size

            for bb in gt:
                x,y,w,h,ign = bb.astype(int)
                if ign:
                    c = (0,0,0)
                else:
                    c = (255,0,0)
                cv2.rectangle(I, (x,y),(x+w,y+h), c, 4)

            cv2.imshow("Hard neg. mining", I)
            cv2.waitKey(1)

            if req_neg <= 0 and req_pos <= 0:
                break

        if self.X0 is None:  # uninitialized
            dtype = new_X0[0].dtype
            self.X0 = np.empty((0,)+self.shape, dtype); self.H0 = np.empty(0, np.float32); self.P0 = 1
            self.X1 = np.empty((0,)+self.shape, dtype); self.H1 = np.empty(0, np.float32); self.P1 = 1

        self.X0 = np.concatenate(new_X0+[self.X0])
        self.H0 = np.concatenate(new_H0+[self.H0])
        self.X1 = np.concatenate(new_X1+[self.X1])
        self.H1 = np.concatenate(new_H1+[self.H1])

    @property
    def n_pos(self):
        return self.X1.shape[0] if self.X1 is not None else 0

    @property
    def n_neg(self):
        return self.X0.shape[0] if self.X0 is not None else 0

    def prune(self, theta):
        self.H0, self.X0, p0 = reject_samples(self.H0, self.X0, theta)
        self.P0 *= p0
        self.H1, self.X1, p1 = reject_samples(self.H1, self.X1, theta)
        self.P1 *= p1
        self.logger.debug(f"Prunning: p0 = {p0:0.3f}, p1 = {p1:0.3f}")
        self.logger.debug(f"Negative rejection rate  {1-self.P0:0.5f}")
        self.logger.debug(f"Positive rejection rate  {1-self.P1:0.5f}")

    def get_positive(self):
        return self.X1, self.H1, self.P1

    def get_negative(self):
        return self.X0, self.H0, self.P0
