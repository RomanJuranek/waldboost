""" Waldboost package

Support for training WaldBoost detectors

Functions
---------
train                       Train new detector
detect, detect_multiple     Detect objects on image
load, load_model            Load model from a file
save, save_model            Save model to file

Classes
-------
DTree
Learner
Model
Pool

Example
-------
import waldboost as wb
from waldboost.utils import fake_data_generator

# Prepare parameters
training_images = fake_data_generator()
chns = wb.default_channel_opts
shape = (14,14,1)  # size is arbitrary, but number of channels must correspond to what is in chns
model = wb.Model(shape, chns)

# Train the model
stats = wb.train(model, training_images)
# stats contain training loss and class probs for each training round

# Draw next image from generator and detect
image,*_ = next(training_images)
bbs,scores = wb.detect(image, model)

# bbs and scores contain locations of detected objects
"""


import logging
from collections import defaultdict
from typing import List

import numpy as np
from pkg_resources import resource_filename

from . import bbox, channels, fpga, nb_channels
from .channels import grad_hist, grad_mag
from .model import Model
from .samples import SamplePool
from .training import BasicRejectionSchedule, DTree, Learner


with open(resource_filename(__name__, "VERSION"), "r") as f:
    __version__ = f.read().strip()


load = load_model = Model.load


def save_model(model, filename):
    """ Save model to file. See Model.save """
    model.save(filename)


save = save_model


def detect(image, model):
    """ Detect objects in image. See Model.detect """
    return model.detect(image)


def detect_multiple(image, *models:List[Model], channel_opts=None, response_scale=None, separate=False):
    """ Detect objects in image using multiple detectors (with shared channel options)

    Inputs
    ------
    image : ndarray
        Input image with shape (H,W) and np.uint8 dtype
    models : waldboost.Model
        Sequence of models to evaluate
    channel_opts : dict
        Channel options. By default models[0].channel_opts are used, but user
        can specify their own. Usually they do not need to be modified, except
        for channel_opts["n_per_oct"] parameter which controls number of scales
        in pyramid.
    separate : bool
        When True, models are treated as multiple classes

    Outputs
    -------
    dt_boxes : BoxList
        Bounding boxes of detected objects with 'scores' field. When separate=False,
        scores are (N,1) array. separate=True results in scores with shape (N,len(models)).
        dt_boxes can be passed to non_max_suppression (when separate=False) or to
        multi_class_non_max_suppression (when separate=True).

    See also
    --------
    waldboost.detect : Detection with single model
    waldboost.bbox.np_bbox_list_ops.non_max_suppression
    waldboost.bbox.np_bbox_list_ops.multi_class_non_max_suppression

    Notes
    -----
    Calling detect_multiple(image, model) is exactly equivalent to detect(image, model)
    and also to model.detect(image).

    Example
    -------
    Multiple classes:
        model_files = ["face.pbz", "car.pbz"]
        models = [waldboost.load(f) for f in model_files]
        image = imread("image.jpg")  # Need grayscale uint8
        detections = detect_multiple(image, *models, separate=True)

    Single class, multiple detectors:
        model_files = ["scale_0.pbz", "scale_1.pbz"]
        models = [waldboost.load(f) for f in model_files]
        image = imread("image.jpg")  # Need grayscale uint8
        # Modify how channels are calculated (if needed)
        channels = models[0].channel_opts
        channels["n_per_oct"] = 1
        detections = detect_multiple(image, *models, channel_opts=channels, separate=False)

    Also call with individual models as positional arguments (instead of list) is possible:
        detect_multiple(image, model_a, model_b, model_c)
    """
    channel_opts = channel_opts or models[0].channel_opts
    if response_scale is None:
        response_scale = [1] * len(models)  # No scale given -> do not change values
    response_scale = np.array(response_scale, "f")
    n_classes = len(models)
    dt_boxes = []
    for chns, scale in channels.channel_pyramid(image, channel_opts):
        # Generate channel pyramid from the input image
        for k, model in enumerate(models):
            # Evaluate each model on the channels
            r,c,h = model.predict_on_image(chns)
            if r.size:
                boxes = bbox.BoxList(model.get_boxes(r, c, scale))
                if separate:
                    scores = np.zeros((boxes.num_boxes(),n_classes), "f")
                    scores[:,k] = h * response_scale[k]
                else:
                    scores = h.reshape((-1,1)) * response_scale[k]
                boxes.add_field("scores", scores)
                dt_boxes.append(boxes)
    return bbox.np_box_list_ops.concatenate(dt_boxes) if dt_boxes else None


def train(model,
          training_images,
          length=64,
          alpha=0.2,
          min_tp_iou=0.75,
          n_pos=1000,
          n_neg=1000,
          max_depth=2,
          theta_schedule=BasicRejectionSchedule(),
          callbacks=[],
          logger=None):
    """ Train or continue training detection model

    This is a baseline training pipeline. It simply trains a classifier
    of specified length using supplied training images.

    Inputs
    ------
    model : waldboost.Model
        Empty or initialized instance of Model. New stages will be appended to
        the model until desired length is reached.
    training_images : generator
        Generator of training data. Training data are read from the generator as
        X,Y,*_ = next(training_images), where X must be ndarray with shape (H,W,C)
        with C>=1 and X.ndim==3. And Y is ndarray with shape (N,5) specifying
        locations of objects in the image in x,y,w,h,ignore format.
    length : scalar int
        Desired length of the model
    alpha : scalar float
        Maximum false negative rate of the classifier (see [1])
    min_tp_iou : scalar float
        Minimal IOU of object samples with ground truth. Samples with IOU > min_tp_iou
        will be considered as positive
    n_pos, n_neg : scalar int
        Number of positive (resp. negative) samples for stage training
    max_depth : scalar int
        Depth of decision trees in the model
    callbacks : list of functions
        List of functions with signature: callback(model, learner, stage)
    logger : logging.Logger

    Outputs
    -------
    stats : dict
        Training statistics with keys "loss", "p0" and "p1" - all ndarray with
        length corresponding to number of stages in the model.
        loss - value of loss function
        p0   - false positive rate. The ratio of background samples that make
               it to the stage. The value should asymptotically decrease to 0
        p1   - true positive rate. Asymptotically decreases to 1-alpha

    Example
    -------
    # Set parameters
    shape = (14,14,1)
    channel_opts = waldboost.deafult_channel_opts

    # Initialize generator and model
    gen = walbdoost.fake_data_generator()
    model = waldboost.Model(shape, channel_opts)

    # Train model
    waldboost.train(model, gen)
    model.save("model.pbz", compress=True)

    # Inference on an image
    image,_ = next(gen)
    bbs,scores = model.detect(image)

    References
    ----------
    [1] Sochman et al.: Waldboost-learning for time constrained sequential detection, CVPR 2005
    """
    logger = logger or logging.getLogger("WaldBoost")

    learner = Learner(alpha=alpha, wh=DTree, max_depth=max_depth, min_samples_leaf=10)
    pool = SamplePool(min_tp=n_pos, min_fp=n_neg, min_tp_iou=min_tp_iou)

    stats = defaultdict(list)
    for stage in range(1,length+1):
        logger.info(f"Training stage {stage}")
        pool.update(model, training_images)
        X0,H0 = pool.get_false_positives()
        X1,H1 = pool.get_true_positives()
        loss,p0,p1 = learner.fit_stage(model, X0, H0, X1, H1, theta=theta_schedule(stage, learner.P0))
        for cb in callbacks:
            cb(model, learner, stage)
        stats["loss"].append(loss)
        stats["p0"].append(p0)
        stats["p1"].append(p1)
    return stats
