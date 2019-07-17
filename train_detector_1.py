import logging
import numpy as np
import copy

import waldboost as wb
from waldboost import fpga
from datasets import cgt_lp
from waldboost.utils import draw_detections
from waldboost.nb_channels import grad_hist_4
import cv2


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s] %(message)s")
logger = logging.getLogger("WaldBoost")

training_images = cgt_lp.one_row_plates(40/12, augment=True)
test_images = cgt_lp.one_row_plates(40/12, augment=False)
test_im,_ = next(test_images)

channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 1,
    "target_dtype": np.int32,
    "channels": [ grad_hist_4 ],
    }

shape = (12,40,4)

def show_cb(model, learner, stage):
    # pylint: disable=no-member
    logging.info(f"p0 = {learner.P0}")
    if learner.P0 < 0.05:
        I = draw_detections(test_im, model)
        cv2.imshow("x", I)  
        cv2.waitKey(10)

M = wb.Model(shape, channel_opts)

stats = wb.train(M,
                training_images,
                alpha=0.1,
                length=128,
                callbacks=[show_cb],
                n_pos=1000,
                n_neg=20000,
                theta_schedule=wb.BasicRejectionSchedule((4,64), 1e-6),
                max_depth=2,
                logger=logger)
