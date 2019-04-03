import math
import numpy as np
from random import uniform, gauss
import bbox


def bb_to_xys(bb, f=3):
    x,y,w,h = bb
    return x+0.5*w, y+0.5*h, f*math.log2(w)


def bb_distance(bb0, bb1):
    a,b,c = bb_to_xys(bb0)
    u,v,w = bb_to_xys(bb1)
    return math.sqrt((u-a)**2 + (v-b)**2 + (w-c)**2)


def bb_overlap_distance(bb0, bb1):
    x0,y0,w0,h0 = bb0
    x1,y1,w1,h1 = bb1
    xa = max(x0,x1)
    ya = max(y0,y1)
    xb = min(x0+w0,x1+w1)
    yb = min(y0+h0,y1+h1)
    #print(xa,xb,ya,yb)
    if xa > xb or ya > yb:
        return 1
    i = (xb-xa) * (yb-ya)
    u = (w0*h0) + (w1*h1) - i
    #print(i,u)
    return 1 - (i/u)



def distance_matrix(dt, gt, metric=bb_overlap_distance):
    return np.array( [[metric(g,d) for d in dt] for g in gt] )


def match(dt, gt=None, allow_multiple=True, metric=bb_overlap_distance):
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
    dist = distance_matrix(dt, gt[...,:4], metric=metric)
    ign_flag = gt[:,4].astype(np.bool) if has_ign_flag else np.zeros(n_gt,np.bool)
    # Todo use linear sum assignment when allow_multiple is false - distance of unassigned dt will be set to default_dist
    if allow_multiple:
        dt_dist = np.min(dist, axis=0)
        dt_ign = ign_flag[np.argmin(dist, axis=0)]
        gt_dist = np.min(dist, axis=1)
    else:
        raise NotImplementedError

    return dt_dist, dt_ign, gt_dist


def read_bbgt(filename, lbls={}, ilbls={}, ar_range=None, ar=1, resize=1):
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
            x,y,w,h = bb = tuple(map(int, elms[1:5]))
            ar = w/h
            ign = bool(int(elms[10]))
            if lbl in lbls:
                if ar_range is not None:
                    a0,a1 = ar_range
                    ign = not (a0 <= ar <= a1) or ign
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


def nms(bbs, score, min_group=1, distance=bb_overlap_distance, max_dist=0.7):
    idx = np.argsort(score)[::-1]
    groups = {}
    suppressed = np.zeros(idx.size, np.bool)
    for i,idx_i in enumerate(idx[:-1]):
        if suppressed[idx_i]: continue
        groups[i] = [idx_i]
        for j,idx_j in enumerate(idx[i+1:]):
            if suppressed[idx_j]: continue
            if distance(bbs[idx_i], bbs[idx_j]) < max_dist:
                groups[i].append(idx_j)
                suppressed[idx_j] = True

    res_bbs = []
    res_score = []
    for gid, group in groups.items():
        res_bbs.append(np.mean(bbs[group], axis=0))
        res_score.append(np.max(score[group]))

    return np.array(res_bbs), np.array(res_score)
