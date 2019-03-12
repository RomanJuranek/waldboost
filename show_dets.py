import glob
from random import shuffle
import cv2
import numpy as np
import waldboost
import pickle
from tensorflow.keras.models import load_model
from itertools import cycle

with open("model8.pkl","rb") as f:
    model = pickle.load(f)

verifier = load_model("model8_verifier.h5")

model["opts"]["pyramid"]["n_per_oct"] = 4

#data = "/mnt/data/Datasets/LP_Zoom/dataset_elite/img/*.jpg"
#data = "/mnt/matylda1/juranek/Datasets/CAMEA/LicensePlatesDataset/testing/images/*.jpg"
data = "/mnt/cgt/home/LP_Detection/dataset/test/img/*.jpg"
img_fs = glob.glob(data)
shuffle(img_fs)

for f in cycle(img_fs):
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    u,v = image.shape
    image = cv2.resize(image, (v//2,u))

    bbs,_ = waldboost.detect(image, model, verifier)
    bbs0, _ = waldboost.detect(image, model)

    bbs = bbs.astype("i").tolist()
    bbs0 = bbs0.astype("i").tolist()
    bbs_nms,_ = cv2.groupRectangles(bbs, 1, 0.5)

    im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for x,y,w,h in bbs0:
        cv2.rectangle(im, (x,y),(x+w,y+h), (0,128,128), 1)

    for x,y,w,h in bbs:
        cv2.rectangle(im, (x,y),(x+w,y+h), (0,255,255), 1)

    crops = []
    for x,y,w,h in bbs_nms:
        cv2.rectangle(im, (x,y),(x+w,y+h), (64,255,64), 2)
        crops.append( cv2.resize(image[y:y+h,x:x+w],(80,20)) )

    if crops:
        crops = np.hstack(crops)
        cv2.imshow("LPS", crops)

    cv2.imshow("DETS", im)

    cv2.waitKey()
