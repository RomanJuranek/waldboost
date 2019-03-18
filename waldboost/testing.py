import logging
import numpy as np
from .detector import detect
from .groundtruth import match, bb_overlap


logger = logging.getLogger(__name__)


def process_images(testing_images, model, verifier=None):
    dts = []
    gts = []
    for img, gt in testing_images:
        dt, score = detect(img, model)
        dt_overlap, dt_ign, gt_overlap = match(dt, gt, bb_overlap)
        dts.append( list(zip(dt, score, dt_overlap, dt_ign)) )
        gts.append( list(zip( gt, gt_overlap )) )

    return gts, dts


def compute_roc(dt, gt, min_overlap=0.7, n_thresholds=1000):
    # Gather all responses and determine their range
    H = np.array( [d[1] for d in dt] )
    h0 = min(H)
    h1 = max(H)

    # generate evaluation thresholds - regularly spaced in range of responses
    T = np.linspace(h0, h1, n_thresholds)

    # Initial counts of tp, fp and fn
    n_tp = np.zeros(n_thresholds)
    n_fp = np.zeros(n_thresholds)
    n_fn = np.zeros(n_thresholds)

    # Number of images (required since the function report FPPI)
    n_imgs = len(dt)

    # Go through the thresholds
    for k,t in enumerate(T):
        # Go through detections and count true positives and false posifives
        for _, score, overlap, ign in dt:
            tp = score>=t and ~ign and overlap>=min_overlap
            fp = score>=t and ~ign and overlap<min_overlap
            n_tp += tp.sum()
            n_fp += fp.sum()
        # Go through annotated objects and count false negatives
        for gt, overlap in gt:
            ign = gt[...,4]
            fn = score>=t and ~ign and overlap<min_overlap
            n_fn += fn.sum()

    # Detection rate is the ration of true positives in true positives and false negatives (i.e. total number of objects)
    det_rate = n_tp / (n_tp + n_fn)
    # FPPI is average nubmer of false positives on each image
    fppi_rate = n_fp / n_imgs

    return 1-det_rate, fppi_rate, T
