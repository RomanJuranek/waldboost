import cv2
import logging
import numpy as np
from .detector import detect
from .groundtruth import match
from sklearn import metrics
import bbx


logger = logging.getLogger(__name__)


def process_images(testing_images, model, verifier=None):
    results = {}
    for img, gt, filename in testing_images:
        dt, score, confidence = detect(img, model, verifier)
        results[filename] = {
            "dt": np.array(dt), "score": score, "confidence": confidence, "gt": np.array(gt)
        }
    return results


def nms_basic(dt, score, confidence):
    mask = score>0
    return bbx.nms(dt[mask,...], score[mask], min_overlap=0.2)


def nms_with_verification(dt, score, confidence):
    mask = np.logical_and(score>0, confidence>0.1)
    return bbx.nms(dt[mask,...], score[mask], min_overlap=0.2)


def compute_roc(results, nms_func, metric=bb_overlap_distance, max_dist=0.3):
    Y = []
    score = []
    ignore = []
    gt_missed = 0
    gt_total = 0

    for f,r in results.items():
        gt = r["gt"]
        dt_nms, score_nms = nms_func(r["dt"], r["score"], r["confidence"])
        dt_dist, dt_ign, gt_dist = match(dt_nms, gt, metric=metric)
        has_ign_flag = gt.shape[1] == 5
        ign_flag = gt[:,4].astype(np.bool) if has_ign_flag else np.zeros(n_gt,np.bool)
        Y.append(dt_dist < max_dist)  # distance to some gt is less than threshold -> tp
        score.append(score_nms)  # Detection score
        ignore.append(dt_ign)
        gt_missed += np.logical_and(gt_dist>=max_dist, ~ign_flag).sum()
        gt_total += (~ign_flag).sum()

    Y = np.concatenate(Y)
    score = np.concatenate(score)
    ignore = np.concatenate(ignore)

    Y = Y[~ignore]
    score = score[~ignore]
    det_rate = 1-(gt_missed / gt_total)
    n_imgs = len(results.keys())

    print(gt_missed, gt_total, det_rate)

    bins = np.linspace(score.min(), score.max(), 1000)
    tp,_ = np.histogram(score[Y==1], bins)
    fp,_ = np.histogram(score[Y==0], bins)
    fppi = (fp.sum() - np.cumsum(fp)) / n_imgs
    miss = 1-(1-(np.cumsum(tp) / tp.sum())) * det_rate

    return fppi, miss, bins
