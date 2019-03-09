import logging
import glob
from itertools import cycle
from random import shuffle
import cv2
import numpy as np

from waldboost import groundtruth, training, samples


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Define data
# /home/ijuranek/matylda1
img_fs = sorted(glob.glob("/mnt/matylda1/juranek/Datasets/CAMEA/LicensePlatesDataset/training/images/*.jpg"))
gt_fs = sorted(glob.glob("/mnt/matylda1/juranek/Datasets/CAMEA/LicensePlatesDataset/training/gt/*.txt"))

def image_generator(img_fs, gt_fs):
    fs = list(zip(img_fs, gt_fs))
    if not fs:
        return
    logger.debug(f"Generating from list of {len(fs)} images")
    shuffle(fs)
    for img_f, gt_f in cycle(fs):
        im = cv2.imread(img_f, cv2.IMREAD_GRAYSCALE)
        h,w = im.shape
        im = cv2.resize(im, (w//2,h))
        gt = groundtruth.read_bbgt(gt_f, lbls={"LP"})
        gt[:,0] /= 2
        gt[:,2] /= 2
        logger.debug(f"Loaded {img_f} with {len(gt)} objects")
        yield im, gt

training_images = image_generator(img_fs, gt_fs)


# Define options
alpha = 0.05
T = 512
n_pos = 10000
n_neg = 30000
shape = (10,40)
name = "model8"

model = {
    "opts": {
        "shape": shape,
        "channels": [
            ("mag", {"norm_rad": 0})
        ],
        "pyramid": {
            "shrink": 2,
            "n_per_oct": 8,
            "smooth": 0,
        }
    },
    "classifier": []
}

samples = samples.SamplePool(training_images, shape, n_pos=n_pos, n_neg=n_neg)

ftr_params = training.feature_params(shape, max_dist=10)

for t in range(T):
    logger.info(f">>> Training stage {t}/{T}")
    samples.update(model)
    pos_samples = samples.get_positive()
    neg_samples = samples.get_negative()
    theta = None if t%2==1 else -np.inf
    params,thr,hs,theta = training.fit_stage(neg_samples, pos_samples, ftr_params, alpha, theta)

    _,H,_ = neg_samples
    p = np.sum(H > theta) / H.size
    if theta > -np.inf and p > 0.95:
        logger.info(f"Neg probability too high {p:.2f} (require < 0.95). Forcing theta to -inf")
        theta = -np.inf

    stage = params, thr, hs, theta
    model["classifier"].append(stage)

    samples.prune(theta)

logger.info("Saving model")
import pickle
with open(name + ".pkl","wb") as f:
    pickle.dump(model, f)


logger.info("Updating training set before training verifier")
samples.min_neg = 50000
samples.update(model)

logger.info("Saving data")
data1 = samples.get_positive()
data0 = samples.get_negative()
with open(name + "_data.pkl","wb") as f:
    pickle.dump([data0, data1], f)

# train verification mdoel
logger.info("Training verification model")
from waldboost import verification
X0,H0,_ = data0
X1,H1,_ = data1
M = verification.model(shape+(1,))
verification.train(M, X0, H0, X1, H1)

logger.info("Saving verification model")
M.save(name + "_verifier.h5")
