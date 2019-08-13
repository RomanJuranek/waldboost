"""
WaldBoost detector training

Example
-------
This shows an example 
    > waldboost-train -d dataset.json -c license_plate --ignore car --shape=40,12 --alpha=0.1 --length=128 -o model.pb


Notes
-----
Models trained with this tool can be used with waldboost-detect.
"""


import argparse
import json
import logging

import cv2
import imgaug
import numpy as np

import waldboost as wb


def parse_arguments():
    parser = argparse.ArgumentParser(description='WaldBoost Detector Training')
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("-o", "--output", type=str, default="model.pb", help="Output model file")
    # --length
    # --alpha
    # --max-depth
    # --max-tp-dist
    # --n-pos
    # --n-neg
    # --shape=40,12
    # -c --class name
    # --ignore name
    # -q Quiet. Log only critical stuff
    # 
    return parser.parse_args()


class DataGenerator:
    """ Dataset specified in JSON

    structure is:
    
    "classes":{"license_plate":0, "car":1, ...}
    "annotations": [
        {"filename": "image.jpg", "boxes": [[x,y,w,h],[...]], "classes": [c,...], "ignore":[...]},
        {...},
        ...
    ]
    """
    def __init__(self, filename, classes):
        pass
    def __iter__(self):
        yield None


if __name__ == "__main__":
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("WaldBoost")

    # Initialize image source and draw one image for visualization
    training_images = DataGenerator(args.dataset, args.classes)

    # Channel options
    channel_opts = {
        "shrink": 2,
        "n_per_oct": 8,
        "smooth": 0,
        "target_dtype": np.int32,
        "channels": [ wb.nb_channels.grad_hist_4 ],
        }

    # Detector window shape - size of smallest detectable object
    shape = (8,40,4)

    M = wb.Model(shape, channel_opts)

    stats = wb.train(M,
                    training_images,
                    alpha=0.05,
                    length=64,
                    callbacks=[],
                    n_pos=1000,
                    n_neg=1000,
                    theta_schedule=wb.BasicRejectionSchedule((4,32), 1e-5),
                    max_depth=2,
                    logger=logger)

    