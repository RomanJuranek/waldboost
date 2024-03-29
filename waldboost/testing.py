"""
Support for detector testing
"""

import logging
import random
from itertools import islice
from typing import Sequence, Any

import bbx
import numpy as np
import sklearn.metrics
from bbx import Boxes

import waldboost as wb


class Evaluator:
    def __init__(self):
        self.clear()
    def clear(self):
        self.eval_data = dict()
    def add_ground_truth(self, idx, boxes:Boxes, shape):
        if idx not in self.eval_data:
            self.eval_data[idx] = dict()
        self.eval_data[idx].update(gt=boxes, shape=shape)
    def add_detections(self, idx, boxes:Boxes):
        if idx not in self.eval_data:
            self.eval_data[idx] = dict()
        self.eval_data[idx].update(dt=boxes)
    def evaluate(self, match_iou_threshold=0.5, dt_iou_threshold=0.5, min_gt_area=0, min_gt_area_in_image=1, normalize_ar=None):
        labels = []
        scores = []
        ignored = []
        for img_result_dict in self.eval_data.values():
            # Get ground truth info
            gt_boxes = img_result_dict["gt"]
            h,w = img_result_dict["shape"]
            img_box = bbx.Boxes([0,0,w,h])
            gt_ignore = np.logical_or.reduce([
                gt_boxes.get_field("ignore") != 0,
                gt_boxes.area() < min_gt_area,
                ~bbx.boxes_in_window(gt_boxes, img_box, min_overlap=min_gt_area_in_image)
                ])
            
            dt_boxes = bbx.non_max_suppression(img_result_dict["dt"], iou_threshold=dt_iou_threshold)
            dt_scores=dt_boxes.get_field("scores")

            if normalize_ar is not None:
                dt_boxes = bbx.set_aspect_ratio(dt_boxes, normalize_ar)
                gt_boxes = bbx.set_aspect_ratio(gt_boxes, normalize_ar)

            iou = bbx.iou(gt_boxes, dt_boxes)
            
            if iou.shape[0] > 0:
                assigned_gt = iou.argmax(axis=0)
                ign = gt_ignore[assigned_gt]
                tp = (iou.max(axis=0) > match_iou_threshold)[~ign]
                score = dt_scores[~ign]
            else:
                tp = np.zeros(len(dt_boxes), np.bool)
                score = dt_scores

            ignored.append(gt_ignore)
            labels.append(tp)
            scores.append(score)

        y_true = np.concatenate(labels)
        scores = np.concatenate(scores)
        ignored = np.concatenate(ignored)

        p,r,t = sklearn.metrics.precision_recall_curve(y_true, scores)
        
        eval_dict = dict(
            precision=p.tolist(), recall=r.tolist(), threshold=t.tolist(),
            auc = sklearn.metrics.auc(r, p),
            iou_threshold=match_iou_threshold,
            n_eval=(ignored==0).sum(),
            n_ign=(ignored!=0).sum()
            )
        return eval_dict


def random_iterator(seq:Sequence[Any], maxlen=None) -> Any:
    """
    Iterate over random elements of the sequence.
    """
    if not hasattr(seq, "__len__") or not hasattr(seq, "__getitem__"):
        raise TypeError("Sequence must be indexable")
    N = len(seq)
    order = list(range(N))
    random.shuffle(order)
    for i,j in enumerate(cycle(order)):
        if maxlen is not None and i > maxlen:
            return
        yield seq[j]


def evaluate_model(testing_images, *model, num_images=None, shuffle=False):
    """
    Test the given model on testing images and return evaluation structure
    """
    if num_images is None:
        if hasattr(testing_images, "__len__"):
            num_images = len(testing_images)
        else:
            raise ValueError("Require num_images with infinite dataset")
    
    if shuffle:
        testing_images = random_iterator(testing_images)
    imgs = islice(testing_images, num_images)

    E = Evaluator()
    logging.info(f"Running model on {num_images} images")
    for idx,(gt,dt,shape) in enumerate(detect_on_images(imgs, *model), start=1):
        E.add_ground_truth(idx, gt, shape)
        E.add_detections(idx, dt)
        if idx % 20 == 0:
            logging.info(f"{idx}")

    if num_images != idx:
        logging.warning(f"Requested test on {num_images} but only {idx} images were given in dataset.")

    return E


def detect_on_images(images, *model, gt_key="groundtruth_boxes"):
    empty_boxes = Boxes(np.empty((0,4)), ignore=[])
    for data_dict in images:
        image = data_dict.get("image")
        gt_boxes = data_dict.get(gt_key, empty_boxes)
        dt_boxes = wb.detect(image, *model)
        yield gt_boxes, dt_boxes, image.shape[:2]
