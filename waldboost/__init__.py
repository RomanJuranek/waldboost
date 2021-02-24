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
from bbx.boxes import Boxes

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


def save_model(model:Model, filename):
    """ Save model to file. See Model.save """
    model.save(filename)


save = save_model


def detect(image:np.ndarray, *models:Model, channel_opts:dict=None, response_scale=None) -> Boxes:
    """
    Detect objects in image using multiple detectors (with shared channel options)

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
    respose_scale: ndarray, list
        List of scale factors for each model. Responses of models are multiplied by
        this factor when stored in the result. This can be useful when different models
        have different ranges of output (due to differences in their training).
        Say model0 has range [-5,5], model1 has range [-1,1], then response_scale=[1,5]
        will equalize the ranges and the result can be used safely in non maxumum suppression.

    Outputs
    -------
    dt_boxes : Boxes
        Bounding boxes of detected objects with 'scores' and 'label' fields.
        The 'scores' represent score for each detection, 'label' is the index
        of model emiting the detection

    See also
    --------
    Model.detect

    Notes
    -----
    Calling detect(image, model) is equivalent to model.detect(image) except for
    label filed in the resulting boxes    
    """
    channel_opts = channel_opts or models[0].channel_opts
    if response_scale is None:
        response_scale = [1] * len(models)  # No scale given -> do not change values
    response_scale = np.array(response_scale, "f")
    if response_scale.size != len(models):
        raise ValueError("Wrong response_scale parameter")
    dt_boxes = []
    for chns, scale in channels.channel_pyramid(image, channel_opts):
        # Generate channel pyramid from the input image
        for k, model in enumerate(models):
            # Evaluate each model on the channels
            r,c,h = model.predict_on_image(chns)
            if r.size == 0: continue
            boxes = model.get_boxes(r, c, scale)
            boxes.set_field("scores", h * response_scale[k])
            boxes.set_field("label", np.full(r.size, k, dtype=np.int))
            dt_boxes.append(boxes)
    return bbx.concatenate(dt_boxes, ["scores", "label"])


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
