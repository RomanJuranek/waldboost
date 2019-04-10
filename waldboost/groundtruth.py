import math
import numpy as np
from random import uniform, gauss
import bbx


def match(dt, gt=None):
    """
    Match detections (dt) to ground truth (gt) by using metric.
    """
    has_gt = gt is not None and gt.size > 0

    assert isinstance(dt, np.ndarray)

    n_dt = dt.shape[0]
    n_gt = gt.shape[0]

    if not has_gt or n_dt == 0:
        gt_dist = np.full(n_gt, np.inf)
        dt_dist = np.full(n_dt, np.inf)
        dt_ign = np.zeros(n_dt, np.bool)
        return dt_dist, dt_ign, gt_dist

    n_gt, gt_cols = gt.shape
    has_ign_flag = gt_cols == 5
    assert gt_cols in [4,5], "GT must have 4 or 5 columns"
    dist = 1 - np.array(bbx.dist_matrix(gt, dt, metric=bbx.overlap))
    ign_flag = gt[:,4].astype(np.bool) if has_ign_flag else np.zeros(n_gt,np.bool)
    # Todo use linear sum assignment when allow_multiple is false - distance of unassigned dt will be set to default_dist
    dt_dist = np.min(dist, axis=0)
    dt_ign = ign_flag[np.argmin(dist, axis=0)]
    gt_dist = np.min(dist, axis=1)

    return dt_dist, dt_ign, gt_dist


def read_bbgt(filename, lbls={}, ilbls={}, ar_range=None, ar=1, resize=1):
    """
    Read ground truth from bbGt file.
    See Piotr's Toolbox for details

    returns Nx5 matrix with columns [x,y,w,h, ignore]
    """
    gt = []
    ignore = []
    with open(filename,"r") as f:
        signature = f.readline()
        assert signature.startswith("% bbGt version=3"), "Wrong signature"
        for line in f:
            elms = line.strip().split()
            assert len(elms) == 12, "Invalid file"
            #print(elms)
            lbl = elms[0]
            x,y,w,h = bb = tuple(map(int, elms[1:5]))
            ar = w/h
            ign = bool(int(elms[10]))
            if lbl in lbls:
                if ar_range is not None:
                    a0,a1 = ar_range
                    ign = not (a0 <= ar <= a1) or ign
                bb = bbx.set_aspect_ratio(bb, ar=ar, type=bbx.KEEP_WIDTH)
                bb = bbx.resize(bb, resize)
                gt.append(bb)
                ignore.append(ign)
            elif lbl in ilbls or ign:
                gt.append(bb)
                ign.append(1)
            else:
                # lbl not in lbls nor ilbls
                pass
    if not gt:
        gt = np.empty((0,5),"f")
        ignore = np.empty(0,"f")
    else:
        gt = np.concatenate(gt,axis=0)
        ignore = np.array(ignore,"f")

    return np.hstack([gt,ignore[:,None]])
