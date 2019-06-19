import logging
import numpy as np
import copy

import waldboost as wb
from datasets import cgt_lp


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)-5.5s] %(message)s")
logger = logging.getLogger("WaldBoost")

shape = (16,40,4)

alpha = 0.2
T = 32
n_pos = 1000
n_neg = 50000
name = "test"

channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 1,
    "target_dtype": np.int16,
    "channels": [ (wb.fpga.grad_hist_4,()) ]
    }

training_images = cgt_lp.one_row_plates(40/16)

test_im, test_gt = next(training_images)

M = wb.Model(shape, channel_opts)
L = wb.Learner(alpha=alpha, n_samples=None, wh=wb.DTree, depth=2)
P = wb.Pool(shape, min_tp=n_pos, min_fp=n_neg)
P.max_tp_dist = 0.25


try:
    for stage in range(len(M)+1,T+1):
        logger.info(f"Training stage {stage}")
        P.update(M, training_images, take_tp=stage==1)
        X0,H0 = P.gather_samples(0)
        X1,H1 = P.gather_samples(1)
        L.fit_stage(M, X0, H0, X1, H1)
        logger.info(f"P0 = {L.P0:.5f}")
        logger.info(f"P1 = {L.P1:.5f}")
        show_detections(M, test_im)
except:
    logging.error("Woo!!! Something went wrong.")

M.save(f"{name}.pb")
