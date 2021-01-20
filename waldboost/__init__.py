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
boxes = wb.detect(image, model)

# boxes contain locations and scores of detected objects
rects = boxes.get()  # rects is (N,4) array with [ymin,xmin,ymax,xmax] coordinates
"""


import logging

import numpy as np
from pkg_resources import resource_filename
import bbx

from . import channels, samples, groundtruth
from .model import Model
from .samples import SamplePool
from .training import BasicRejectionSchedule, DTree, Learner


def _set_version():
    with open(resource_filename(__name__, "VERSION"), "r") as f:
        return f.read().strip()


__version__ = _set_version()


load = load_model = Model.load


def save_model(model, filename):
    """ Save model to file. See Model.save """
    model.save(filename)


save = save_model


def detect(image,
           model,
           iou_threshold=0.2,
           score_threshold=-10):
    """ Detect objects in image. See Model.detect """
    boxes = model.detect(image)
    if boxes:
        boxes = bbx.non_max_suppression(boxes, iou_threshold=iou_threshold, score_threshold=score_threshold)
    return boxes


def detect_multiple(image,
                    *models,
                    channel_opts=None,
                    response_scale=None,
                    separate=False,
                    iou_threshold=0.2,
                    score_threshold=-10):
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
            boxes = bbox.BoxList(model.get_boxes(r, c, scale))
            if separate:
                scores = np.zeros((boxes.num_boxes(),n_classes), "f")
                scores[:,k] = h * response_scale[k]
            else:
                scores = h * response_scale[k]
            boxes.add_field("scores", scores)
            dt_boxes.append(boxes)
    dt_boxes = bbox.np_box_list_ops.concatenate(dt_boxes)
    if dt_boxes.num_boxes() > 0:
        nms_func = bbox.non_max_suppression if not separate else bbox.multi_class_non_max_suppression
        dt_boxes = nms_func(dt_boxes, iou_threshold=iou_threshold, score_threshold=score_threshold)
    return dt_boxes


def train(model,
           training_images,
           learner=None,
           pool=None,
           length=64,
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
    learner : Learner
        Learner that adds new stages to the model. If none supplied, an instance
        of Learner with default parameters is created. This instance is also
        returned from the function in order to enable training continuation.
    pool : SamplePool
        Object that holds training samples.
    length : int
        Total number of stages to train, i.e. len(model) == length after train
        finish.
    theta_schedule : callable
        A function that decides (based on current false positive rate and stage number)
        if rejection threshold for the current stage should be learned. If no
        function is given, an instance of BasicRejectionSchedule is used.
    callbacks : list of functions
        List of functions with signature: callback(model, learner, stage). After training
        each stage, all functions are executed. This may be useful for monitoring
        loss function or displaying detection results on a testing image.
    logger : logging.Logger

    Outputs
    -------
    learner : Learner
        Instance of Learner that was used for training. In case when the
        learner was passed to the function this is the same instance.

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
    logger = logger or logging.getLogger("training")

    if len(model) >= length:
        return

    learner = learner or Learner()

    if  len(model) != len(learner):
        raise RuntimeError("Model length and learner length are not consistent")

    if len(model) > 0:
        logger.info(f"{len(model)} stages are already present, continuing")

    pool = pool or SamplePool()

    for stage in range(len(model), length):
        logger.info(f"Training stage {stage}")
        pool.update(model, training_images)
        X0,H0 = pool.get_false_positives()
        X1,H1 = pool.get_true_positives()
        loss,p0,p1 = learner.fit_stage(model, X0, H0, X1, H1, theta=theta_schedule(stage, learner.false_positive_rate))
        logger.debug(f"Loss: {loss:0.5f}, fpr: {p0:0.5f}, tpr: {p1:0.5f}")
        for cb in callbacks:
            cb(model, learner, stage)

    return learner


def train_softcascade(shape,
                      channel_opts,
                      length,
                      training_images,
                      callbacks):

    logging.warning("Do not use this function")

    casc_thr = -1.0
    pool = SamplePool(
        min_tp=4000,
        min_fp=20000,
        min_tp_iou=0.75,
        max_tp_candidates=5,
        max_fp_candidates=50,
    )

    model = Model(shape, channel_opts)
    pool.update(model, training_images)

    for stage, stage_length in enumerate(length):
        print(f"Stage {stage}: length={stage_length}")
        learner = Learner(max_depth=2)
        pool.print_stats()
        for w in range(stage_length):
            print(f"weak={w}")
            pool.update(model, training_images)
            X0,H0 = pool.get_false_positives()
            X1,H1 = pool.get_true_positives()
            loss,*_ = learner.fit_stage(model, X0, H0, X1, H1, theta=-np.inf)
            print(f"loss={loss:.6f}")
        model.theta = [casc_thr] * stage_length
        for cb in callbacks:
            cb(model, learner, stage)
        if stage == len(length)-1:
            return model
        pool.update(model, training_images)
        for boxes in pool.samples:
            boxes.get_field("scores")[:] = 0
        model = Model(shape, channel_opts)
