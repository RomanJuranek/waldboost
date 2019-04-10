import logging
import numpy as np
import bbx

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


def get_new_samples(chns, scale, gt, shape, classifier, max_pos=100, max_neg=100):
    if classifier is not None:
        r,c,h = forward(chns, shape, classifier)
    else:
        r0,c0,h0 = sample_random(chns, shape)
        r1,c1,h1 = sample_from_bbs(chns, shape, bbx.scale(gt, scale))
        r = np.concatenate([r0,r1])
        c = np.concatenate([c0,c1])
        h = np.concatenate([h0,h1])

    dt = bbs_from_dets(r, c, shape, scale)
    dt_dist, dt_ign, _ = groundtruth.match(dt, gt)

    def take_samples(mask, max_n):
        if np.any(mask):
            idx = np.nonzero(mask)[0]
            if len(idx) > max_n:
                idx = np.random.choice(idx, max_n, replace=False)
            X = gather_samples(chns, r[idx], c[idx], shape)
            H = h[idx]
        else:
            X = np.zeros((0,)+shape, "f")
            H = np.zeros(0, "f")
        return X, H

    fp = np.logical_and(dt_dist>0.8, ~dt_ign)
    X0, H0 = take_samples(fp, max_neg)

    tp = np.logical_and(dt_dist<0.5, ~dt_ign)
    X1, H1 = take_samples(tp, max_pos)

    return X0, H0, X1, H1


class SamplePool:
    def __init__(self, shape, channel_opts, n_neg=1000, n_pos=1000, logger=None):
        self.shape = shape
        self.channel_opts = channel_opts
        self.min_neg = n_neg
        self.min_pos = n_pos
        self.logger = logger or logging.getLogger(__name__)
        self.dtype = channel_opts["target_dtype"]
        self.X0 = np.empty((0,)+self.shape, self.dtype); self.H0 = np.empty(0, np.float32)
        self.X1 = np.empty((0,)+self.shape, self.dtype); self.H1 = np.empty(0, np.float32)

    def update(self, generator, detector):
        req_neg = self.min_neg - self.n_neg
        req_pos = self.min_pos - self.n_pos
        self.logger.info(f"Pool size: {self.n_pos} positive, {self.n_neg} negative;")
        self.logger.info(f"Require: {req_pos} positives, {req_neg} negatives")
        if req_neg <= 0 and req_pos <= 0:
            self.logger.debug("Nothing to update (pool is full)")
            return

        new_X0 = []
        new_H0 = []
        new_X1 = []
        new_H1 = []
        classifier = detector["classifier"]

        for im, gt in generator:
            for chns, scale in channel_pyramid(im, detector["opts"]):
                X0,H0,X1,H1 = get_new_samples(chns, scale, gt, self.shape, classifier, max_pos=min(req_pos,100), max_neg=min(req_neg,100))
                new_X0.append(X0)
                new_H0.append(H0)
                new_X1.append(X1)
                new_H1.append(H1)
                req_neg -= H0.size
                req_pos -= H1.size
                self.logger.debug(f"Sampled {H0.size} negatives and {H1.size} positives")
            if req_neg <= 0 and req_pos <= 0:
                break

            #     if req_neg > 0:
            #         r,c,h = predict_and_sample(chns, detector)
            #         dt = bbs_from_dets(r, c, self.shape, scale)
            #         dt_dist, dt_ign, _ = groundtruth.match(dt, gt)
            #         fp = dt_dist > 0.8
            #         dt_ign = np.logical_and(dt_ign, ~fp)
            #         if np.any(fp):
            #             fp_idx = np.nonzero(fp)[0]
            #             if fp_idx.size > 50:
            #                 fp_idx = np.random.choice(fp_idx, 50)
            #             new_X0.append(gather_samples(chns, r[fp_idx], c[fp_idx], self.shape))
            #             new_H0.append(h[fp_idx])
            #             req_neg -= fp_idx.size
            #
            #         for bb, ign, is_fp in zip(dt, dt_ign, fp):
            #             x,y,w,h = bb.astype(int)
            #             if ign:
            #                 c = (0,0,0)
            #             elif is_fp:
            #                 c = (0,255,0)
            #             else:
            #                 c = (0,0,255)
            #             cv2.rectangle(I, (x,y),(x+w,y+h), c, 1)
            #
            #     if req_pos > 0:
            #         r,c,h = sample_from_bbs(chns, self.shape, gt[:,:4]*scale)
            #         dt = bbs_from_dets(r, c, self.shape, scale)
            #         dt_dist, dt_ign, _ = groundtruth.match(dt, gt)
            #         tp = np.logical_and(dt_dist<0.25, ~dt_ign)
            #         if np.any(tp):
            #             tp = np.nonzero(tp)[0]
            #             new_X1.append(gather_samples(chns, r[tp], c[tp], self.shape))
            #             new_H1.append(h[tp])
            #             req_pos -= tp.size
            #             logging.debug(f"Added {tp.size} pos. samples")
            #
            # for bb in gt:
            #     x,y,w,h,ign = bb.astype(int)
            #     if ign:
            #         c = (0,0,0)
            #     else:
            #         c = (255,0,0)
            #     cv2.rectangle(I, (x,y),(x+w,y+h), c, 4)
            #
            # cv2.imshow("Hard neg. mining", I)
            # cv2.waitKey(1)
            #
            # if req_neg <= 0 and req_pos <= 0:
            #     break

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
        self.H1, self.X1, p1 = reject_samples(self.H1, self.X1, theta)
        self.logger.debug(f"Prunning: p0 = {p0:0.3f}, p1 = {p1:0.3f}")
        return p0, p1
        #self.logger.debug(f"Negative rejection rate  {1-self.P0:0.5f}")
        #self.logger.debug(f"Positive rejection rate  {1-self.P1:0.5f}")

    def get_positive(self):
        return self.X1, self.H1

    def get_negative(self):
        return self.X0, self.H0
