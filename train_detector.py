import logging
import glob
from itertools import cycle
from random import shuffle
import cv2
import numpy as np
from os.path import basename

import waldboost
from waldboost import groundtruth, training, samples
from waldboost.channels import grad_mag, grad_hist
from waldboost.fpga import grad_hist_4, Quantizer
from waldboost.image import random_adjust
from waldboost.utils import save_cache
from waldboost import verification


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)-5.5s] %(message)s")
logger = logging.getLogger("WaldBoost")


# Define data
# /home/ijuranek/matylda1
data = "/mnt/cgt/home/LP_Detection/dataset/train"
#data = "/mnt/matylda1/juranek/Datasets/CAMEA/LicensePlatesDataset/training"
img_fs = sorted(glob.glob(data + "/img/*.jpg"))
gt_fs = sorted(glob.glob(data + "/gt/*.txt"))


def bbgt_image_generator(img_fs, gt_fs, daylight=True, ar=8, ar_range=(8,14)):
    """
    """
    fs = list(zip(img_fs, gt_fs))
    if not fs:
        return
    logger.debug(f"Generating from list of {len(fs)} images")
    shuffle(fs)
    for img_f, gt_f in cycle(fs):
        gt = groundtruth.read_bbgt(gt_f, lbls={"lp"}, target_ar=ar, resize=(1.3,2), ar_range=ar_range)
        if np.all(gt[...,-1]==1):
            logger.debug(f"Skipping {basename(img_f)}")
            continue

        im = cv2.imread(img_f, cv2.IMREAD_GRAYSCALE)
        h,w = im.shape
        im = cv2.resize(im, (w//2,h))
        y0 = int(0.15*h)
        y1 = int(0.85*h)
        im = im[y0:y1,...]

        if daylight is not None:
            if daylight and np.median(im) < 50:
                continue
            if not daylight and np.median(im) > 50:
                continue

        im = random_adjust(im)

        gt[:,0] /= 2
        gt[:,2] /= 2
        gt[:,1] -= y0

        logger.debug(f"Loaded {basename(img_f)} with {len(gt)} objects")
        yield im, gt

alpha = 0.25
T = 64
n_pos = 2000
n_neg = 10000
shape = (10,26,4)
name = "cgt_lp_0002"


channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 0,
    "target_dtype": np.int16,
    "channels": [ (grad_hist_4,()) ]
    #"channels": [ ]
    }

# q = Quantizer((-4,4),8)

training_images = bbgt_image_generator(img_fs, gt_fs, daylight=True)
model = waldboost.Model(shape, channel_opts, alpha=alpha, n_pos=n_pos, n_neg=n_neg, wh=waldboost.training.DTree, depth=3)
history = model.fit(training_images, T=T)
save_cache(model.as_dict(), f"{name}.pkl")

#
# logger.info("Updating training set before training verifier")
# tr_samples.min_neg = 50000
# tr_samples.update(model)
#
# logger.info("Saving data")
# data1 = tr_samples.get_positive()
# data0 = tr_samples.get_negative()
# # save_cache([data0, data1], name+"_data.pkl")
#
# # train verification mdoel
# logger.info("Training verification model")
# X0,H0,_ = data0
# X1,H1,_ = data1
# M = verification.model_cnn(shape)
# verification.train(M, X0, H0, X1, H1)
# logger.info("Saving verification model")
# M.save(name + "_verifier.h5")
