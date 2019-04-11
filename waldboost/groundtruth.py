import numpy as np
import bbx


def match(dt, gt):
    """
    Match detections (dt) to ground truth (gt) by using metric.
    """

    try:
        dist = 1 - bbx.dist_matrix(dt, gt, metric=bbx.overlap)
    except:
        print(gt)
        print(dt)
        raise ValueError

    n_gt, n_dt = dist.shape

    if dist.size == 0:
        # Either gt or dt is empty
        dt_dist = np.full(n_gt, 1, np.float32)
        dt_ign = np.zeros(n_gt,np.bool)
        gt_dist = np.full(n_dt, 1, np.float32)
        return dt_dist, dt_ign, gt_dist

    # There is at least one dt and one gt
    n_gt, gt_cols = gt.shape
    has_ign_flag = gt_cols == 5
    assert gt_cols in [4,5], "GT must have 4 or 5 columns"

    ign_flag = gt[:,4].astype(np.bool) if has_ign_flag else np.zeros(n_gt,np.bool)
    # Todo use linear sum assignment when allow_multiple is false - distance of unassigned dt will be set to default_dist
    dt_dist = np.min(dist, axis=1)
    dt_ign = ign_flag[np.argmin(dist, axis=1)]
    gt_dist = np.min(dist, axis=0)

    return dt_dist, dt_ign, gt_dist


def read_bbgt(filename, lbls={}, ilbls={}, ar_range=None, target_ar=1, resize=1):
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
            ign = int(elms[10])
            ar = bb[2]/bb[3]
            if lbl in lbls:
                if ar_range is not None:
                    a0,a1 = ar_range
                    ign = not (a0 <= ar <= a1) or ign
                bb = bb + (ign,)
                bb = bbx.set_aspect_ratio(bb, ar=target_ar, type=bbx.KEEP_WIDTH)
                bb = bbx.resize(bb, resize)
                gt.append(bb)
            elif lbl in ilbls or ign:
                gt.append( bb + (1,))
            else:
                # lbl not in lbls nor ilbls
                pass
    if not gt:
        gt = np.empty((0,5),"f")
    else:
        gt = np.concatenate(gt,axis=0)

    return gt
