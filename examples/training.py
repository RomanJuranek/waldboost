""" Detector training example
"""

import cv2
import logging
import numpy as np
import copy

import waldboost as wb
from waldboost.utils import draw_detections, fake_data_generator
from datasets import cgt_lp


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("WaldBoost")

# Initialize image source and draw one image for visualization
training_images = cgt_lp.one_row_plates(40/8,True)
test_im,_ = next(training_images)

# Channel options
channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 1,
    "target_dtype": np.int32,
    "channels": [ wb.nb_channels.grad_hist_4 ],
    }

# Detector window shape - size of smallest detectable object
shape = (8,40,4)

# Callback that shows image and detections
def show_cb(model, learner, stage):
    # pylint: disable=no-member
    logging.info(f"p0 = {learner.P0}")
    test_im,_ = next(training_images)
    if learner.P0 < 0.05:
        I = draw_detections(test_im, model)
        cv2.imshow("x", I)  
        cv2.waitKey(10)

M = wb.Model(shape, channel_opts)

stats = wb.train(M,
                training_images,
                alpha=0.05,
                length=64,
                callbacks=[show_cb],
                n_pos=4000,
                n_neg=100000,
                theta_schedule=wb.BasicRejectionSchedule((4,32), 1e-5),
                max_depth=3,
                logger=logger)
