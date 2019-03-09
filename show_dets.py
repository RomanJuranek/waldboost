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

model["opts"]["pyramid"]["n_per_oct"] = 8

#data = "/mnt/data/Datasets/LP_Zoom/dataset_elite/img/*.jpg"
#data = "/mnt/matylda1/juranek/Datasets/CAMEA/LicensePlatesDataset/testing/images/*.jpg"
data = "/mnt/cgt/home/LP_Detection/dataset/test/img/*.jpg"
img_fs = glob.glob(data)
shuffle(img_fs)

for f in cycle(img_fs):
    image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    u,v = image.shape
    image = cv2.resize(image, (v//2,u))
    bbs, score = waldboost.detect(image, model, verifier)
    bb_list = [ bb.astype("i").tolist() for bb,h in zip(bbs,score) if h > 0]
    bbs ,w = cv2.groupRectangles(bb_list, 2, 0.5)
    #bbs = bb_list
    im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x,y,w,h in bbs:
        cv2.rectangle(im, (x,y),(x+w,y+h), (0,255,0), 2)
    cv2.imshow("DETS", im)
    cv2.waitKey(1)
