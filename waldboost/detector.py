"""
"""

import numpy as np
import cv2


def channel_pyramid(image, opts):
    pyr_opts = opts["pyramid"]
    chns_opts = opts["channels"]

    assert pyr_opts["shrink"] in [1,2], "Shrink factor can be only 1 or 2"

    factor = 2**(-1/pyr_opts["n_per_oct"])

    smooth_H = np.array((1,2,1), np.float) / 4
    norm_H = np.array((1,2,3,4,5,4,3,2,1), np.float)
    norm_H /= norm_H.sum()
    deriv_H = np.array((-1,0,1), np.float) / 2

    base_image = image
    m,n =  opts["shape"]
    while True:
        h, w = base_image.shape
        for i in range(pyr_opts["n_per_oct"]):
            s = factor ** i
            nw, nh = int((w*s)/2)*2, int((h*s)/2)*2
            if nh < 2*m or nw < 2*n:
                return
            real_scale = nw / image.shape[1]
            im = cv2.resize(base_image, (nw, nh), cv2.INTER_LINEAR)
            im = im.astype(np.float32) / 256
            gx = cv2.filter2D(im, cv2.CV_32F, deriv_H[None,:])
            gy = cv2.filter2D(im, cv2.CV_32F, deriv_H[:,None])
            mag = cv2.magnitude(gx, gy)
            mag = cv2.resize(mag, (nw//2, nh//2))
            #norm = cv2.sepFilter2D(mag, cv2.CV_32F, norm_H, norm_H)
            #chn = mag / (norm + 1e-3)
            chn = mag
            yield chn, real_scale/2
        #print("downscale")
        base_image = cv2.resize(base_image, (w//2, h//2))


def np_forward(chns, m, n, ftr, hs, thr, theta):
    u,v = chns.shape
    idx = np.arange(np.int32(max(u-m,0)*max(v-n,0)))
    rs = idx % (u-m)
    cs = idx // (u-m)
    Hs = np.zeros_like(rs, np.float32)
    for f, h, t, th in zip(ftr, hs, thr, theta):
        r0, c0, r1, c1 = f
        fs = chns[rs+r0,cs+c0] - chns[rs+r1,cs+c1]
        Hs += h[(fs > t)*1]
        if th == -np.inf:
            continue
        mask = Hs >= th
        rs = rs[mask]
        cs = cs[mask]
        Hs = Hs[mask]
    return rs, cs, Hs


def np_classifier(c):
    n = len(c)
    ftr = np.empty( (n,4), np.int32 )
    hs = np.empty( (n,2), np.float32 )
    thr = np.empty( n, np.float32 )
    theta = np.empty( n, np.float32 )
    for t, (f, t0, h, th) in enumerate(c):
        ftr[t,...] = f
        hs[t,...] = h
        thr[t] = t0
        theta[t] = th
    return ftr, hs, thr, theta


def forward(chns, detector):
    m,n = detector["opts"]["shape"]
    ftr, hs, thr, theta = np_classifier(detector["classifier"])
    return np_forward(chns, m, n, ftr, hs, thr, theta)


def detect(image, detector, verifier=None):
    from .samples import gather_samples
    bbs = []
    H = []
    shape = m, n = detector["opts"]["shape"]
    ftr, hs, thr, theta = np_classifier(detector["classifier"])
    for chns, scale in channel_pyramid(image, detector["opts"]):
        r, c, h = np_forward(chns, m, n, ftr, hs, thr, theta)
        if r.size > 0:
            if verifier is not None:
                print(f"Verifying {r.size} detections ({(h>0).size})")
                X = gather_samples(chns, r, c, shape)
                Y = verifier.predict([X[:,:,:,None], h]) # None is for conversion (N,H,W) -> (N,H,W,1)
                verified = np.nonzero(Y > 0.9)[0]
                r = r[verified]
                c = c[verified]
                h = h[verified]
            if r.size > 0:
                bbs.extend( bbs_from_dets(r, c, shape, scale) )
                H.extend( h )
    return np.array(bbs,"f"), np.array(H,"f")


def bbs_from_dets(r, c, shape, scale):
    m,n = shape
    return np.array([(c,r,n,m) for r,c in zip(r,c)], np.float) / scale
