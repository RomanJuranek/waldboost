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
import numpy as np
from collections import defaultdict
from pkg_resources import resource_filename

from .training import DTree, Learner
from .samples import Pool
from .model import Model
from .channels import grad_mag, grad_hist
from . import fpga, channels, nb_channels


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


def detect_multiple(image, *models, channel_opts=None, response_scale=None, separate=False):
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
        When True, results for each model are returned as separate arrays,
        otherwise they are merged together. Usually, when models are intended
        to detect different classes, separate=True. When models detect the same
        class (e.g. in different scales, or different kinds of class), then
        separate=False.

    Outputs
    -------
    bbs, scores : default_dict or ndarray
        Bounding boxes of detected objects. In case of separate=True it is an
        instance of default_dict where bbs[k] and scores[k], correspond to results
        of models[k]. Otherwise it is ndarray with results merged form all models.
        The results correspond to that of waldboost.detect.

    See also
    --------
    waldboost.detect : Detection with single model

    Notes
    -----
    In case of a single model (i.e. detect_multiple(image, model)) the result
    is exactly equivalent to detect(image, model) and also to model.detect(image).

    Example
    -------
    Multiple classes:
        model_files = ["face.pbz", "car.pbz"]
        models = [waldboost.load(f) for f in model_files]
        image = imread("image.jpg")  # Need grayscale uint8
        bbs, scores = detect_multiple(image, *models, separate=True)
        # bbs[k] and scores[k] correspond to results of models[k]

    Single class, multiple detectors:
        model_files = ["scale_0.pbz", "scale_1.pbz"]
        models = [waldboost.load(f) for f in model_files]
        image = imread("image.jpg")  # Need grayscale uint8
        # Modify how channels are calculated (if needed)
        channels = models[0].channel_opts
        channels["n_per_oct"] = 1
        bbs, scores = detect_multiple(image, channel_opts=channels, eparate=False, *models)
        # bbs and scores are merged results for all models

    Also call with individual models as positional arguments (instead of list)
    is possible:
        detect_multiple(image, model_a, model_b, model_c, separate=True)
    """
    channel_opts = channel_opts or models[0].channel_opts
    if response_scale is None:
        response_scale = [1] * len(models)  # No scale given -> do not change values
    response_scale = np.array(response_scale, "f")

    bbs = defaultdict(list)
    scores = defaultdict(list)
    for chns, scale in channels.channel_pyramid(image, channel_opts):
        # Generate channel pyramid from the input image
        for k, model in enumerate(models):
            # Evaluate each model on the channels
            r,c,h = model.predict_on_image(chns)
            bbs_k = model.get_bbs(r,c,scale)
            if separate:
                bbs[k].append(bbs_k)
                scores[k].append(h)
            else:
                bbs[0].append(bbs_k)
                scores[0].append(h)
    for k in bbs:  # k is model key to bbs and scores
        # Merge groups of bounding boxes
        bbs_k = [x for x in bbs[k] if x.size]
        scores_k = [x for x in scores[k] if x.size]
        if bbs_k:
            bbs[k] = np.concatenate(bbs_k)
            scores[k] = np.concatenate(scores_k) * response_scale[k]
        else:
            bbs[k] = np.empty((0,4))
            scores[k] = np.empty(0)
    if not separate:
        bbs, scores = bbs[0], scores[0]

    return bbs, scores


default_channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 0,
    "target_dtype": np.float32,
    "channels": [ grad_mag ]
    }


def _class_prob_cb(model, learner, stage):
    print(f"{learner.P0}")


def train(model,
          training_images,
          length=64,
          alpha=0.2,
          max_tp_dist=0.25,
          n_pos=1000,
          n_neg=1000,
          max_depth=2,
          target_p0=1e-5,
          callbacks=[_class_prob_cb],
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
    max_tp_dist : scalar float
        Maximal IOU of object samples with ground truth (samples with IOU < max_tp_dist)
        will be considered as positive
    n_pos, n_neg : scalar int
        Number of positive (resp. negative) samples for stage training
    max_depth : scalar int
        Depth of decision trees in the model
    target_p0 : scalar
        Condition for stopping rejection of training samples. If 0 < stop_sampling < 1
        (usually close to zero, like 1e-5), rejection stops when p0 (false positive
        probability) reaches below this threshold. When stop_sampling>=1, rejection
        stops at the specified stage.
    callbacks : list of functions
        List of functions with signature: callback(model, stage, learner)
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

    def theta(n, p0):
        assert target_p0 > 0, "target_p0 must be positive"
        if 0 < target_p0 < 1:
            theta = None if p0 > target_p0 else -np.inf
        else:
            theta = None if n < target_p0 else -np.inf
        return theta

    logger = logger or logging.getLogger("WaldBoost")

    learner = Learner(alpha=alpha, wh=DTree, max_depth=max_depth, min_samples_leaf=10)
    pool = Pool(model.shape, min_tp=n_pos, min_fp=n_neg)
    pool.max_tp_dist = max_tp_dist

    stats = defaultdict(list)
    for stage in range(1,length+1):
        logger.info(f"Training stage {stage}")
        pool.update(model, training_images)
        X0,H0 = pool.gather_samples(0)
        X1,H1 = pool.gather_samples(1)
        loss,p0,p1 = learner.fit_stage(model, X0, H0, X1, H1, theta=theta(stage, learner.P0))
        for cb in callbacks:
            cb(model, learner, stage)
        stats["loss"].append(loss)
        stats["p0"].append(p0)
        stats["p1"].append(p1)

    return stats
