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


def cost_matrix(bbs0, bbs1, cost=bb_distance):
    raise NotImplementedError


def partition(bbs, gt=None, dist_thr=0):
    """
    Partition bounding boxes to two groups by proximity to a ground truth bounding boxes.

    If no gt is specified, partition evaluates to false.
    """
    if gt is None or gt.size == 0:
        return np.zeros(bbs.shape[0], np.bool)
    dist = [[bb_distance(g,d) for d in bbs] for g in gt]
    dist = np.array(dist)
    dist = np.min(dist, axis=0)
    return dist < dist_thr


def read_bbgt(filename, lbls=None, ilbls=None, ar=1, resize=1):
    """
    Read ground truth from bbGt file.
    See Piotr's Toolbox for details
    """
    gt = []
    with open(filename,"r") as f:
        signature = f.readline()
        assert signature.startswith("% bbGt version=3"), "Wrong signature"
        for line in f:
            elms = line.split()
            lbl = elms[0]
            if lbl in lbls:
                bb = tuple(map(int, elms[1:5]))
                bb = bbox.set_aspect_ratio(bb, ar=ar, type=bbox.KEEP_WIDTH)
                bb = bbox.resize(bb, resize)
                gt.append(bb)
    return np.array(gt)
