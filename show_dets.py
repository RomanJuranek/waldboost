import glob
from random import shuffle
import cv2
import numpy as np
import waldboost
from tensorflow.keras.models import load_model
from itertools import cycle
from waldboost.utils import load_cache
from waldboost.image import random_adjust
import bbx
import time


name = "cgt_lp_0001"
model = load_cache(name + ".pkl")
verifier = load_model(name + "_verifier.h5")

#model["opts"]["pyramid"]["n_per_oct"] = 6

#data = "/mnt/data/Datasets/LP_Zoom/dataset_elite/img/*.jpg"
#data = "/mnt/matylda1/juranek/Datasets/CAMEA/LicensePlatesDataset/testing/images/*.jpg"
data = "/mnt/cgt/home/LP_Detection/dataset/test/img/*.jpg"
#data = "/mnt/data/digisaw/Honkalahti_20180324/*.png"
#data = "/mnt/data/digisaw/mechanical_damages/04/*.bmp"
#data = "/mnt/data/Traffic/BTech/libalpr/events-cam5-01/19/*.jpg"
img_fs = sorted(glob.glob(data))
shuffle(img_fs)

for f in img_fs[:]:
    print(f)
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    u,v = image.shape
    image = cv2.resize(image, (v//2,u))
    #image = random_adjust(image)
    u,v = image.shape

    t0 = time.monotonic()
    bbs,score,confidence = waldboost.detect(image, model, verifier)
    tm = time.monotonic() - t0
    print(f"Duration {1000*tm:0.1f}ms")

    #print(bbs.shape, score.shape, confidence.shape)
    mask = np.logical_and(score>0, confidence>0.5)
    bbs_nms,_ = bbx.nms(bbs[mask,...], score[mask], min_group=2, min_overlap=0.2)
    #
    im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    crops = []
    for x,y,w,h in bbs_nms.astype("i"):
        cv2.rectangle(im,(x,y),(x+w,y+h),(64,255,64), 2)
        # crop_bb = bbox.set_aspect_ratio((x,y,w,h), 1, bbox.KEEP_WIDTH)
        # crop_bb = bbox.resize(crop_bb, 1.1)
        # x,y,w,h = np.array(crop_bb).astype("i")
        # x0 = np.clip(x,0,v)
        # y0 = np.clip(y,0,u)
        # x1 = np.clip(x+w,0,v)
        # y1 = np.clip(y+h,0,v)
        # crops.append( cv2.resize(image[y0:y1,x0:x1],(160,160)) )
    #
    # if crops:
    #     crops = np.hstack(crops)
    #     cv2.imshow("LPS", crops)
    #
    cv2.imshow("DETS", im)
    cv2.waitKey()

for i,c in enumerate(crops):
    cv2.imwrite(f"lp_{i:04d}.png", c)
