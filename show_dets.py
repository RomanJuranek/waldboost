import glob
from random import shuffle
import cv2
import numpy as np
import waldboost as wb
from itertools import cycle
import bbx
import time
from skimage.measure import block_reduce

name = "test"
model = wb.Model.load(f"{name}.pb")

#data = "/mnt/data/Datasets/LP_Zoom/dataset_elite/img/*.jpg"
#data = "/mnt/matylda1/juranek/Datasets/CAMEA/LicensePlatesDataset/testing/images/*.jpg"
#data = "/mnt/cgt/home/LP_Detection/dataset/test/img/*.jpg"
#data = "/mnt/data/digisaw/Honkalahti_20180324/*.png"
#data = "/mnt/data/digisaw/mechanical_damages/04/*.bmp"
#data = "/mnt/data/Traffic/BTech/libalpr/events-cam5-01/19/*.jpg"
data = "/mnt/data/Traffic/QS-PRG/*.jpg"
#data= "/mnt/matylda1/juranek/Datasets/INRIAPerson/test/pos/*.png"

img_fs = sorted(glob.glob(data))
shuffle(img_fs)

for f in img_fs:
    print(f)
    image = cv2.imread(f, cv2.IMREAD_COLOR)

    t0 = time.monotonic()
    bbs,score = model.detect(np.atleast_3d(block_reduce(image, (3,3,3), np.mean).astype("u1")))
    tm = time.monotonic() - t0
    print(f"Duration {1000*tm:0.1f}ms; {len(score)} detections")
    bbs[:,:4] *= 3

    mask = score > 1
    bbs_nms,score_nms = bbx.nms(bbs[mask,...], score[mask], min_group=4, min_overlap=0.1)

    for x,y,w,h in bbs_nms.astype("i"):
        cv2.rectangle(image,(x,y),(x+w,y+h),(64,255,64), 2)

    cv2.imshow("detections", image)
    cv2.waitKey()
