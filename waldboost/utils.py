"""
Dirty code
"""


import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import cm

from . import groundtruth
import bbx


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
                    gt_color=(255,0,0),
                    vmin=None,vmax=None): 
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
    if image.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw gt_boxes
    if gt_boxes is not None:
        for x1,y1,x2,y2 in gt_boxes.get().astype("i"):
            cv2.rectangle(img, (x1,y1), (x2,y2), gt_color, thickness=gt_thickness)

    # Draw dt_boxes
    if dt_boxes:
        scores = dt_boxes.get_field("scores")

        if vmin is None:
            vmin = scores.min()
        if vmax is None:
            vmax = scores.max()

        N = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = cm.plasma      

        dt_boxes = bbx.sort_by_field(dt_boxes, "scores", descending=True)
        for (x1,y1,x2,y2),score in zip(dt_boxes.get().astype("i"), dt_boxes.get_field("scores") ):
            clr = (255*np.array(colors(N(score)))).astype("u1")
            clr = tuple(map(int, clr))
            cv2.rectangle(img, (x1,y1), (x2,y2), clr, thickness=dt_thickness)
    
    return img[...,::-1]


def fake_data_generator():
    while True:
        image = np.zeros( (256,256), "f" )
        gt = [  ]
        n_objects = np.random.randint(2)
        for _ in range(n_objects):
            w = np.random.randint(30,60)
            x = np.random.randint(256-w)
            y = np.random.randint(256-w)
            i = np.random.uniform(0.2,1)
            image[y:y+w,x:x+w] += i
            gt.append( [y-5,x-5,y+w+5,x+w+5] )
        image += np.random.rand(*image.shape) * 0.3*np.random.rand()
        image = (np.clip(image, 0, 1)*255).astype("u1")
        gt = np.array(gt,"f") if gt else np.empty((0,4))
        gt_boxes = groundtruth.bbox_list(gt, format=groundtruth.RectFormat.YXYX)
        yield dict(image=np.atleast_2d(image), groundtruth_boxes=gt_boxes)


class ShowImageCallback():
    """Callback that shows image and detections"""
    def __init__(self, image, gt_boxes, max_fpr=0.05):
        self.image = image
        self.gt = gt_boxes
        self.max_fpr = max_fpr
    def __call__(self, model, learner, stage):
        if learner.false_positive_rate < self.max_fpr:
            dt_boxes = model.detect(self.image)
            I = draw_detections(self.image, dt_boxes, self.gt, gt_thickness=3, gt_color=(255,0,0))
            cv2.imshow("Testing image", I)  # pylint: disable=no-member
            cv2.waitKey(20)  # pylint: disable=no-member


def class_prob_callback(model, learner, stage):
    print(f"Stage {stage}:")
    print(f"\tp0 = {learner.false_positive_rate:.5f}; p1 = {learner.true_positive_rate:.5f}")
