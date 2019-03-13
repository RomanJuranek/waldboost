"""
"""

import numpy as np
import cv2
from .channels import channel_pyramid


def np_forward(chns, m, n, ftr, hs, thr, theta):
    u,v,_ = chns.shape
    idx = np.arange(np.int32(max(u-m,0)*max(v-n,0)))
    rs = idx % (u-m)
    cs = idx // (u-m)
    chs = np.empty_like(rs)
    Hs = np.zeros_like(rs, np.float32)
    for f, h, t, th in zip(ftr, hs, thr, theta):
        # r0, c0, ch0 = f
        # chs[:] = ch0
        # fs = chns[rs+r0,cs+c0,chs]
        # Hs += h[(fs > t).astype(np.uint8)]
        Hs += eval_dstump(chns, rs, cs, chs, f, h, t)
        if th == -np.inf:
            continue
        mask = Hs >= th
        rs = rs[mask]
        cs = cs[mask]
        chs = chs[mask]
        Hs = Hs[mask]
    return rs, cs, Hs


def eval_dstump(chns, rs, cs, chs, ftr, hs, thr):
    r0, c0, ch0 = ftr
    chs[:] = ch0
    fs = chns[rs+r0,cs+c0,chs]
    return hs[(fs > thr).astype(np.uint8)]


def np_classifier(c):
    n = len(c)
    ftr = np.empty( (n,3), np.int32 )
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
    m,n,_ = detector["opts"]["shape"]
    ftr, hs, thr, theta = np_classifier(detector["classifier"])
    return np_forward(chns, m, n, ftr, hs, thr, theta)


def detect(image, detector, verifier=None):
    from .samples import gather_samples

    shape = m,n,_ = detector["opts"]["shape"]
    ftr, hs, thr, theta = np_classifier(detector["classifier"])

    X = []
    R = []
    C = []
    S = []
    H = []

    # Loop over the channel pyramid and gather results
    for chns, scale in channel_pyramid(image, detector["opts"]):
        r, c, h = np_forward(chns, m, n, ftr, hs, thr, theta)
        mask = h > 1
        r = r[mask]
        c = c[mask]
        h = h[mask]
        if r.size > 0:
            if verifier is not None:
                X.append( gather_samples(chns, r, c, shape ))
            R.append( r )
            C.append( c )
            S.append( [scale]*r.size )
            H.append( h )

    if not R:
        return np.array([]), np.array([])

    R = np.concatenate(R)
    C = np.concatenate(C)
    S = np.concatenate(S)
    H = np.concatenate(H)

    print("R", R.shape)
    print("C", C.shape)
    print("S", S.shape)
    print("H", H.shape)

    if verifier is not None:
        X = np.concatenate(X)
        print("X", X.shape)
        Y = verifier.predict([X, H])
        mask = np.nonzero(Y > 0.1)[0]
        R = R[mask]
        C = C[mask]
        S = S[mask]
        H = H[mask]

    return bbs_from_dets(R, C, shape, S), H


def bbs_from_dets(r, c, shape, scale):
    m = shape[0]
    n = shape[1]
    if isinstance(scale, np.ndarray):
        return np.array([(c,r,n,m) for r,c in zip(r,c)], np.float) / scale[:,None]
    return np.array([(c,r,n,m) for r,c in zip(r,c)], np.float) / scale
