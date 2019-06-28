""" Waldboost package
"""


import logging
from .training import DTree, Learner
from .samples import Pool
from .model import Model
from .channels import grad_mag, grad_hist
from . import fpga, channels
import numpy as np


# with open(path.join(path.abspath(path.dirname(__file__)),"VERSION"),"r") as f:
#     __version__ = f.read().strip()


load_model = Model.load


def save_model(model, filename):
    """ Save model. See Model.save """
    model.save(filename)


def detect(image, model):
    """ Detect objects in image. See Model.detect """
    return model.detect(image)


deafult_channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 0,
    "target_dtype": np.float32,
    "channels": [ grad_mag ]
    }


def train(model,
          training_images,
          length=64,
          alpha=0.2,
          max_tp_dist=0.25,
          n_pos=1000,
          n_neg=1000,
          max_depth=2,
          callbacks=[],
          logger=None):
    """ Train or continue training detection model

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
    callbacks : list of functions
    logger : logging.Logger

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
    pool = Pool(model.shape, min_tp=n_pos, min_fp=n_neg)
    pool.max_tp_dist = max_tp_dist

    for stage in range(1,length+1):
        logger.info(f"Training stage {stage}")
        pool.update(model, training_images)
        X0,H0 = pool.gather_samples(0)
        X1,H1 = pool.gather_samples(1)
        learner.fit_stage(model, X0, H0, X1, H1)
        for cb in callbacks:
            cb(stage, model, learner)
