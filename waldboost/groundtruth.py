import math
import numpy as np
from random import uniform, gauss
import bbox


def bb_to_xys(bb, f=20):
    x,y,w,h = bb
    return x+0.5*w, y+0.5*h, f*math.log2(w)


def bb_proximity(bb0, bb1):
    a,b,c = bb_to_xys(bb0)
    u,v,w = bb_to_xys(bb1)
    return (u-a)**2 + (v-b)**2 + (w-c)**2


def partition(bbs, gt=None, max_tp_dist=50, min_fp_dist=50):
    if gt is None:
        return [], np.arange(bbs.shape[0])
    dist = [[bb_proximity(g,d) for d in bbs] for g in gt]
    dist = np.array(dist)
    dist = np.min(dist, axis=0)
    return np.nonzero(dist<max_tp_dist)[0], np.nonzero(dist>min_fp_dist)[0]


def read_bbgt(filename, lbls=None, ilbls=None):
    gt = []
    with open(filename,"r") as f:
        signature = f.readline()
        assert signature.startswith("% bbGt version=3"), "Wrong signature"
        for line in f:
            elms = line.split()
            lbl = elms[0]
            # print(elms[1:5])
            if lbl in lbls:
                bb = tuple(map(int, elms[1:5]))
                bb = bbox.set_aspect_ratio(bb, ar=8, type=bbox.KEEP_WIDTH)
                bb = bbox.resize(bb, 1.2)
                gt.append(bb)
    return np.array(gt)
