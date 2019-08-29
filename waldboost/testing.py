"""
Support for detector testing

This requires TF Object detection API installed
https://github.com/tensorflow/models
"""

import logging

import numpy as np

import waldboost as wb
from object_detection.utils.object_detection_evaluation import \
    ObjectDetectionEvaluator

from . import bbox


def detection_dict(dt_boxes):
    """
    Convert BoxList to detection dict
    """
    return {
        "detection_boxes": dt_boxes.get(),
        "detection_scores": dt_boxes.get_field("scores"),
        "detection_classes": np.ones(dt_boxes.num_boxes(), "i"),
    }


def evaluate_model(model, testing_images, category_index=None, **detect_kws):
    """ Test the given model on testing images and return evaluation structure

    Inputs
    ------
    model : list of wb.Model
    testing_images : generator
    category_index : dict
    detect_kws :
        Allowed keyword arguments are:
        * channel_opts
        * response_scale
        * separate
        * iou_threshold,
        * score_threshold
        See wb.detect_multiple for details
    """

    if isinstance(model, wb.Model):
        model = [model]
    
    if not isinstance(model, list):
        raise ValueError("Model must be waldboost.Model or list of waldboost.Model")

    if category_index is None:
        category_index = [{"id":1,"name":"object"}]

    E = ObjectDetectionEvaluator(categories=category_index,
                                 evaluate_precision_recall=True,
                                 matching_iou_threshold=0.5,
                                 evaluate_corlocs=True)
    

    for idx,(im,gt_boxes,filename) in enumerate(testing_images):
        logging.debug(f"Processing {filename}")
        dt_boxes = wb.detect_multiple(im, *model, **detect_kws)
        dt_dict = detection_dict(dt_boxes)
        gt_dict = {
            "groundtruth_boxes": gt_boxes.get(),
            "groundtruth_classes": np.ones(gt_boxes.num_boxes(),"i"),
            "groundtruth_difficult": gt_boxes.get_field("ignore").flatten(),
        }
        E.add_single_ground_truth_image_info(idx, gt_dict)
        E.add_single_detected_image_info(idx, dt_dict)
    
    return E.evaluate()
