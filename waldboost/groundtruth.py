import math
import numpy as np
from random import uniform, gauss
import bbox


def bb_to_xys(bb, f=20):
    x,y,w,h = bb
    return x+0.5*w, y+0.5*h, f*math.log2(w)


def bb_distance(bb0, bb1):
    a,b,c = bb_to_xys(bb0)
    u,v,w = bb_to_xys(bb1)
    return math.sqrt((u-a)**2 + (v-b)**2 + (w-c)**2)


def bb_overlap_distance(bb0, bb1):
    raise NotImplementedError


def distance_matrix(dt, gt, metric=bb_distance):
    return np.array( [[metric(g,d) for d in dt] for g in gt] )


def match(dt, gt=None, allow_multiple=True, metric=bb_distance, default_dist=np.inf):
    """
    Match detections (dt) to ground truth (gt) by using metric.
    """
    has_gt = gt is not None and gt.size > 0

    assert isinstance(dt, np.ndarray)

    n_dt = dt.shape[0]

    if not has_gt or n_dt == 0:
        gt_dist = []
        dt_dist = np.full(n_dt, np.inf)
        dt_ign = np.zeros(n_dt, np.bool)
        return dt_dist, dt_ign, gt_dist

    n_gt, gt_cols = gt.shape
    has_ign_flag = gt_cols == 5
    assert gt_cols in [4,5], "GT must have 4 or 5 columns"
    dist = distance_matrix(dt, gt[...,:4], metric=metric)
    ign_flag = gt[...,4].astype(np.bool) if has_ign_flag else np.zeros(n_gt,np.bool)
    # Todo use linear sum assignment when allow_multiple is false - distance of unassigned dt will be set to default_dist
    if allow_multiple:
        dt_dist = np.min(dist, axis=0)
        dt_ign = ign_flag[np.argmin(dist, axis=0)]
        gt_dist = np.min(dist, axis=1)
    else:
        raise NotImplementedError

    return dt_dist, dt_ign, gt_dist


def read_bbgt(filename, lbls={}, ilbls={}, ar=1, resize=1):
    """
    Read ground truth from bbGt file.
    See Piotr's Toolbox for details

    returns Nx5 matrix with columns [x,y,w,h, ignore]
    """
    gt = []
    with open(filename,"r") as f:
        signature = f.readline()
        assert signature.startswith("% bbGt version=3"), "Wrong signature"
        for line in f:
            elms = line.strip().split()
            assert len(elms) == 12, "Invalid file"
            #print(elms)
            lbl = elms[0]
            bb = tuple(map(int, elms[1:5]))
            ign = bool(int(elms[10]))
            if lbl in lbls:
                bb = bbox.set_aspect_ratio(bb, ar=ar, type=bbox.KEEP_WIDTH)
                bb = bbox.resize(bb, resize)
                gt.append(bb + (ign,))
            elif lbl in ilbls or ign:
                gt.append(bb + (1,))
                pass
            else:
                # lbl not in lbls nor ilbls
                pass
    gt = np.array(gt,"f")
    if gt.size == 0:
        gt = np.empty((0,5),"f")
    return gt
