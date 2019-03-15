import logging
import glob
from itertools import cycle
from random import shuffle
import cv2
import numpy as np
from os.path import basename

from waldboost import groundtruth, training, samples
from waldboost.channels import grad_mag, grad_hist
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
def bbgt_image_generator(img_fs, gt_fs):
    """
    """
    fs = list(zip(img_fs, gt_fs))
    if not fs:
        return
    logger.debug(f"Generating from list of {len(fs)} images")
    shuffle(fs)
    for img_f, gt_f in cycle(fs):
        im = cv2.imread(img_f, cv2.IMREAD_GRAYSCALE)
        h,w = im.shape
        im = cv2.resize(im, (w//2,h))
        im = random_adjust(im)
        gt = groundtruth.read_bbgt(gt_f, lbls={"lp"}, ar=8, resize=1.3)
        if gt.size > 0:
            gt[:,0] /= 2
            gt[:,2] /= 2
        I = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for bb in gt:
            x,y,w,h = bb.astype(int)
            cv2.rectangle(I, (x,y),(x+w,y+h), (64,255,64), 2)
        cv2.imshow("Image",I)
        cv2.waitKey(1)
        logger.debug(f"Loaded {basename(img_f)} with {len(gt)} objects")
        yield im, gt


# Define options
alpha = 0.2
T = 50
n_pos = 20000
n_neg = 50000
shape = (10,40,4)
name = "model9"

model = {
    "opts": {
        "shape": shape,
        "pyramid": {
            "shrink": 2,
            "n_per_oct": 12,
            "smooth": 0,
            "target_dtype": np.uint8,
            "channels": [ (grad_hist, (4,)) ]
        }
    },
    "classifier": []
}

training_images = bbgt_image_generator(img_fs, gt_fs)
tr_samples = samples.SamplePool(training_images, shape, n_pos=n_pos, n_neg=n_neg)
training.fit_model(model, tr_samples, alpha, T, wh=training.DecisionTree)
save_cache(model, name+".pkl")

logger.info("Updating training set before training verifier")
tr_samples.min_neg = 25000
tr_samples.update(model)

logger.info("Saving data")
data1 = tr_samples.get_positive()
data0 = tr_samples.get_negative()
save_cache([data0, data1], name+"_data.pkl")

# train verification mdoel
logger.info("Training verification model")
X0,H0,_ = data0
X1,H1,_ = data1
M = verification.model(shape)
verification.train(M, X0, H0, X1, H1)
logger.info("Saving verification model")
M.save(name + "_verifier.h5")
