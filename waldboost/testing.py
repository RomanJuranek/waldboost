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


def detect(image, model, iou_threshold=0.3, score_threshold=-10):
    """
    Wrapper that produces detection in TF Object Detection API format

    Input
    -----
    image : ndarray
        Input image - whatever accepter by model.detect
    model : list of models

    Output
    ------
    dt_dict : dict
        Dictionary with keys "detection_boxes", "detection_scores" and
        "detection_classes" as required by ObjectDetectionEvaluator
    """
    dt_boxes = wb.detect_multiple(image, *model, separate=False)

    dt_boxes = bbox.np_box_list_ops.non_max_suppression(dt_boxes,
                                                        iou_threshold=iou_threshold,
                                                        score_threshold=score_threshold)
    return {
        "detection_boxes": dt_boxes.get(),
        "detection_scores": dt_boxes.get_field("scores"),
        "detection_classes": np.ones(dt_boxes.num_boxes(), "i"),
    }


def evaluate_model(model, testing_images, label="object"):
    """ Test the given model on testing images and return evaluation structure """
    category_index = [{"id":1,"name":label}]
    E = ObjectDetectionEvaluator(categories=category_index,
                                 evaluate_precision_recall=True,
                                 matching_iou_threshold=0.5,
                                 evaluate_corlocs=True)

    for idx,(im,gt_boxes,filename) in enumerate(testing_images):
        logging.debug(f"Processing {filename}")
        dt_dict = detect(im, model)
        gt_dict = {
            "groundtruth_boxes": gt_boxes.get(),
            "groundtruth_classes": np.ones(gt_boxes.num_boxes(),"i"),
            "groundtruth_difficult": gt_boxes.get("ignore").flatten(),
        }
        E.add_single_ground_truth_image_info(idx, gt_dict)
        E.add_single_detected_image_info(idx, dt_dict)
    
    return E.evaluate()
