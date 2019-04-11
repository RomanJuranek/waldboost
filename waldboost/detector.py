"""
"""


import numpy as np
from .channels import channel_pyramid


def forward(chns, shape, classifier):
    u,v,ch_image = chns.shape
    m,n,ch_cls = shape
    assert ch_image == ch_cls, "Invalid shape"

    idx = np.arange(max(u-m,0)*max(v-n,0), dtype=np.int32)
    rs = idx % (u-m)
    cs = idx // (u-m)
    hs = np.zeros_like(rs, np.float32)

    for weak, theta in classifier:
        hs += weak.predict_on_image(chns, rs, cs)
        if theta == -np.inf:
            continue
        mask = hs >= theta
        rs = rs[mask]
        cs = cs[mask]
        hs = hs[mask]

    return rs, cs, hs


def detect(image, detector, verifier=None):
    from .samples import gather_samples

    shape = m,n,_ = detector["shape"]

    X = []
    R = []
    C = []
    S = []
    H = []

    # Loop over the channel pyramid and gather results
    for chns, scale in channel_pyramid(image, detector["channel_opts"]):
        r, c, h = forward(chns, shape, detector["classifier"])
        if verifier is not None:
            X.append(gather_samples(chns, r, c, shape))
        R.append( r )
        C.append( c )
        S.append( [scale]*r.size )
        H.append( h )

    R = np.concatenate(R)
    C = np.concatenate(C)
    S = np.concatenate(S)
    H = np.concatenate(H)

    if verifier is not None and R.size > 0:
        X = np.concatenate(X, axis=0)
        confidence = verifier.predict([X, H])
        confidence = confidence[...,0]
    else:
        confidence = np.ones_like(H)

    return bbs_from_dets(R, C, shape, S), H, confidence


def bbs_from_dets(r, c, shape, scale):
    m = shape[0]
    n = shape[1]
    if len(r) == 0:
        return np.empty( (0,4), "f")
    if isinstance(scale, np.ndarray):
        res = np.array([(c,r,n,m) for r,c in zip(r,c)], np.float) / scale[:,None]
    else:
        res = np.array([(c,r,n,m) for r,c in zip(r,c)], np.float) / scale
    return np.atleast_2d(res)
