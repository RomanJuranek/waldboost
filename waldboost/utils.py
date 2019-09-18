"""
Dirty code
"""


import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import cm

import waldboost as wb

from . import groundtruth, bbox


def takemin(iterable, objective=None):
    current_min_el = None
    for idx, el in enumerate(iterable):
        current_el = el if objective is None else objective(el)
        if current_min_el is None or current_el < current_min_el:
            current_min_el = current_el
            yield idx, el


def save_cache(data, filename):
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_cache(filename):
    import pickle
    with open(filename, "rb") as f:
        return pickle.load(f)


def montage(X, nrows=4, ncols=4, figsize=None):
    import matplotlib.pyplot as plt
    import numpy as np
    if figsize is None:
        width = 15
        h,w = X.shape[1:]
        f = w*ncols / width
        figsize = width, (h*nrows)/f
        print(figsize)
    fig = plt.figure(figsize=figsize)
    vmax = np.percentile(X,99)
    for i,x in enumerate(X[:nrows*ncols]):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(x, cmap="gray", vmax=vmax)
        plt.axis("off")
    plt.show()


def draw_detections(image,
                      dt_boxes,
                      gt_boxes=None, *,
                      dt_thickness=1,
                      gt_thickness=1,
                      gt_color=(255,0,0)): 
    """ Draw detected objects in image

    Inputs
    ------
    image : ndarray
        Image to draw the boxes in. Gray or BGR uint8 image.
    dt_boxes : BoxList
        Detected boxes with optional fields "scores", "tp_label", "classes".
    gt_boxes : BoxList
        Ground truth boxes
    """
    img = image.copy()
    if image.shape[2] == 1:
        img = cv2.cvtColor(img[...,0], cv2.COLOR_GRAY2BGR)
    
    # Draw gt_boxes
    if gt_boxes is not None:
        for ymin,xmin,ymax,xmax in gt_boxes.get().astype("i"):
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), gt_color, thickness=gt_thickness)

    # Draw dt_boxes
    if dt_boxes.num_boxes() > 0:
        scores = dt_boxes.get_field("scores")
        N = mpl.colors.Normalize(vmin=scores.min()-1e-1, vmax=scores.max())
        colors = cm.plasma

        dt_boxes = bbox.sort_by_field(dt_boxes, "scores", bbox.SortOrder.ASCEND)
        for (ymin,xmin,ymax,xmax),score in zip(dt_boxes.get().astype("i"), dt_boxes.get_field("scores") ):
            clr = (255*np.array(colors(N(score)))[2::-1]).astype("u1")
            clr = tuple(map(int, clr))
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), clr, thickness=dt_thickness)
    
    return img


def fake_data_generator():
    while True:
        image = np.zeros( (256,256), "f" )
        gt = [  ]
        n_objects = np.random.randint(0,4)
        for _ in range(n_objects):
            w = np.random.randint(30,50)
            x = np.random.randint(256-w)
            y = np.random.randint(256-w)
            i =  np.random.uniform(0.2,1)
            image[y:y+w,x:x+w] += i
            gt.append( [y-5,x-5,y+w+10,x+w+10] )
        image += 0.1*np.random.rand(*image.shape)
        image = (np.clip(image, 0,1)*255).astype("u1")
        gt = np.array(gt,"f") if gt else np.empty((0,4))
        gt_boxes = groundtruth.bbox_list(gt)
        yield np.atleast_3d(image), gt_boxes


class ShowImageCallback():
    """Callback that shows image and detections"""
    def __init__(self, image, gt_boxes):
        self.image = image
        self.gt = gt_boxes
    def __call__(self, model, learner, stage):
        if learner.false_positive_rate < 0.05:
            dt_boxes = model.detect(self.image)
            I = draw_detections(self.image, dt_boxes, self.gt, gt_thickness=3, gt_color=(255,0,0))
            cv2.imshow("Testing image", I)  # pylint: disable=no-member
            cv2.waitKey(20)  # pylint: disable=no-member


def class_prob_callback(model, learner, stage):
    print(f"Stage {stage}:")
    print(f"\tp0 = {learner.P0:.5f}; p1 = {learner.P1:.5f}")
