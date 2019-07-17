""" Detector training example
"""

import cv2
import logging
import numpy as np
import copy

import waldboost as wb
from waldboost.utils import draw_detections, fake_data_generator

logger = logging.getLogger("WaldBoost")

# Initialize image source and draw one image for visualization
training_images = fake_data_generator()
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
shape = (16,16,4)

# Callback that shows image and detections
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
                alpha=0.05,
                length=64,
                callbacks=[show_cb],
                n_pos=2000,
                n_neg=4000,
                theta_schedule=wb.BasicRejectionSchedule((4,64), 1e-6),
                max_depth=2,
                logger=logger)
