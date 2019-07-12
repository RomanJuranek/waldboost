import math
import logging
import numpy as np
import bbx

from . import groundtruth


def sample_from_bbs(chns, shape, bbs):
    u,v,_ = chns.shape
    m,n,_ = shape
    map = np.zeros((max(u-m,0),max(v-n,0)), np.bool)
    for bb in bbs:
        x,y,w,h,ign = bb.astype(np.int)
        if ign or not w or not h: continue
        if abs(math.log((m*n) / (h*w))) < 0.2:  # Area of gt and sample must match approximately
            map[y-m:y+h,x-n:x+w] = 1
    r,c = np.nonzero(map)
    return r, c, np.zeros_like(r, np.float32)


def gather_samples(chns, rs, cs, shape):
    #u,v = chns.shape
    m,n,_ = shape
    if rs.size == 0:
        return np.empty((0,)+shape, dtype=chns.dtype)
    X = [ chns[r:r+m,c:c+n,...] for r,c in zip(rs, cs) ]
    return np.array(X)


class Pool:
    def __init__(self, output_shape, min_fp=1000, min_tp=1000, logger=None):
        self.samples = []
        self.output_shape = output_shape
        self.min_fp_dist = 0.9
        self.max_tp_dist = 0.2
        self.min_tp = min_tp
        self.min_fp = min_fp
        self.max_candidates = 100
        self.logger = logger or logging.getLogger("Pool")

    def clear(self):
        self.samples.clear()

    def classify_dt(self, dt, gt):
        dt_dist, dt_ign, _ = groundtruth.match(dt, gt)
        tp = np.logical_and(dt_dist < self.max_tp_dist, ~dt_ign)
        fp = dt_dist > self.min_fp_dist
        return np.flatnonzero(tp), np.flatnonzero(fp)

    def update(self, detector, gen, take_tp=True, take_fp=True):
        self.logger.info("Updating sample in the pool")
        self.samples = [ self.gather_samples(0) + (0,),
                         self.gather_samples(1) + (1,)]
        pruned_samples = []
        for x,h,y in self.samples:
             _h,mask = detector.predict(x)
             if np.any(mask):
                 pruned_samples.append(  (x[mask,...], _h[mask], y) )
        self.samples = pruned_samples
        req_tp = self.require(1) and take_tp
        req_fp = self.require(0) and take_fp
        if not req_tp and not req_fp:
            return
        self.logger.info("Sampling new data")
        for image, gt, *_ in gen:
            req_tp = self.require(1) and take_tp
            if req_tp and gt.size == 0:
                continue;
            logging.info(f"Req TP: {self.require(1)}, Req FP: {self.require(0)}")
            for chns, scale, (r,c,h) in detector.scan_channels(image):
                n_locations = len(r)
                if n_locations > self.max_candidates and not detector:  # limit the number of candidates
                    k = np.random.choice(n_locations, self.max_candidates)
                    r,c,h = r[k],c[k],h[k]

                if not detector and take_tp and gt.shape[0]:
                    scale_gt = gt.copy()
                    scale_gt[:,:4] *= scale
                    r_tp,c_tp,h_tp = sample_from_bbs(chns, detector.shape, scale_gt)
                    dt_tp = detector.get_bbs(r_tp, c_tp, scale)
                    real_tp,_ = self.classify_dt(dt_tp, gt)
                    if len(real_tp) > 20:
                         real_tp = np.random.choice(real_tp, 20)
                    r = np.concatenate( [r, r_tp[real_tp]] )
                    c = np.concatenate( [c, c_tp[real_tp]] )
                    h = np.concatenate( [h, h_tp[real_tp]] )

                dt = detector.get_bbs(r, c, scale)
                tp, fp = self.classify_dt(dt, gt)

                if take_fp and self.require(0):
                    new_fp = gather_samples(chns, r[fp], c[fp], self.output_shape)
                    self.append(new_fp, h[fp], 0)

                if take_tp and self.require(1):
                    new_tp = gather_samples(chns, r[tp], c[tp], self.output_shape)
                    self.append(new_tp, h[tp], 1)

            req_tp = self.require(1) and take_tp
            req_fp = self.require(0) and take_fp
            if not req_tp and not req_fp:
                break

        else:
            logging.warning("Not enough training images")

    def append(self, X, H, Y):
        self.samples.append( (X, H, Y) )

    def gather_samples(self, c):
        if not self.size(c):
            return np.empty((0,)+self.output_shape), np.empty(0)
        X = np.concatenate( [x for x,h,y in self.samples if y == c] )
        H = np.concatenate( [h for x,h,y in self.samples if y == c] )
        return X, H

    def size(self, c):
        return sum( h.size for _,h,y in self.samples if y == c )

    def require(self, c):
        v = self.min_fp if c==0 else self.min_tp
        return max(v-self.size(c), 0)
