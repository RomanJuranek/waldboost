import math
import numpy as np
from random import uniform, gauss
import bbox


def bb_to_xys(bb, f=32):
    x,y,w,h = bb
    return x+0.5*w, y+0.5*h, f*math.log2(w)


def bb_distance(bb0, bb1):
    a,b,c = bb_to_xys(bb0)
    u,v,w = bb_to_xys(bb1)
    return math.sqrt((u-a)**2 + (v-b)**2 + (w-c)**2)


def bb_overlap(bb0, bb1):
    raise NotImplementedError


def cost_matrix(dt, gt, cost=bb_distance):
    return np.array( [[cost(g,d) for d in bbs] for g in gt] )


def partition(bbs, gt=None, dist_thr=0):
    """
    Partition bounding boxes to two groups by proximity to a ground truth bounding boxes.

    If no gt is specified, partition evaluates to false.
    """
    n_bbs = bbs.shape[0]

    if gt is None or gt.size == 0:
        return np.zeros(n_bbs, np.bool)

    n_gt, gt_cols = gt.shape
    has_ign_flag = gt_cols == 5
    assert gt_cols in [4,5], "GT must have 4 or 5 columns"

    dist = cost_matrix(bbs, gt[...,:4], bb_distance)
    ign_flag = gt[...,4] if has_ign_flag else np.zeros(n_bbs)

    bb_dist = np.min(dist, axis=0)
    bb_ign = ign_flag[np.argmin(dist, axis=0)]

    gt_dist = np.min(dist, axis=1)

    return bb_dist<dist_thr, bb_ign, gt_dist<dist_thr


def read_bbgt(filename, lbls=None, ilbls=None, ar=1, resize=1):
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
            elms = line.split()
            assert len(elms) == 11, "Invalid file"
            lbl = elms[0]
            bb = tuple(map(int, elms[1:5]))
            ign = bool(elms[10])
            if lbl in lbls and not ign:
                bb = bbox.set_aspect_ratio(bb, ar=ar, type=bbox.KEEP_WIDTH)
                bb = bbox.resize(bb, resize)
                gt.append(bb + (0,))
            elif lbl in ilbls or ign:
                gt.append(bb + (1,))
                pass
            else:
                # lbl not in lbls nor ilbls
                pass
    return np.array(gt)
