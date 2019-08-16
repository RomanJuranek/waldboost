"""
Dirty code
"""

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


def draw_detections(img, model, nms=False, min_score=-10):
    import numpy as np
    import matplotlib as mpl
    from matplotlib import cm
    from . import bbox
    #from .verification import detect_and_verify
    import cv2
    #cv2.namedWindow("x", cv2.WINDOW_NORMAL)

    boxes = model.detect(img)

    I = cv2.cvtColor(img[...,0], cv2.COLOR_GRAY2BGR)

    if not boxes:
        return I

    if nms:
        boxes = bbox.np_box_list_ops.non_max_suppression(boxes, iou_threshold=0.2, score_threshold=min_score)
    else:
        boxes = bbox.np_box_list_ops.filter_scores_greater_than(boxes, min_score)

    if not boxes or boxes.num_boxes() == 0:
        return I

    scores = boxes.get_field("scores")
    v0, v1 = scores.min()-1e-1, scores.max()

    N = mpl.colors.Normalize(vmin=v0, vmax=v1)
    colors = cm.plasma

    boxes = bbox.sort_by_field(boxes, "scores", bbox.SortOrder.ASCEND)
    for (ymin,xmin,ymax,xmax),score in zip(boxes.get().astype("i"), boxes.get_field("scores") ):
        clr = (255*np.array(colors(N(score)))[2::-1]).astype("u1")
        clr = tuple(map(int, clr))
        cv2.rectangle(I, (xmin,ymin), (xmax,ymax), clr, thickness=4 if nms else 1)

    return I


def fake_data_generator():
    import numpy as np
    while True:
        image = np.zeros( (256,256), "f" )
        gt = []
        n_objects = np.random.randint(10)
        for _ in range(n_objects):
            w = np.random.randint(20,100)
            x = np.random.randint(256-w)
            y = np.random.randint(256-w)
            i =  np.random.uniform(0.2,1)
            image[y:y+w,x:x+w] += i
            gt.append( [x-5,y-5,w+10,w+10,0] )
            image += 0.05*np.random.randn(*image.shape)
        image = (np.clip(image, 0,1)*255).astype("u1")
        yield np.atleast_3d(image), np.atleast_2d(gt).astype("f")


import cv2


class ShowImageCallback():
    """Callback that shows image and detections"""
    def __init__(self, image, **kws):
        self.image = image
        self.kws = kws
    def __call__(self, model, learner, stage):
        if learner.P0 < 0.05:
            I = draw_detections(self.image, model, **self.kws)
            cv2.imshow("x", I)  # pylint: disable=no-member
            cv2.waitKey(10)  # pylint: disable=no-member


def class_prob_callback(model, learner, stage):
    print(f"Stage {stage}:")
    print(f"\tp0 = {learner.P0:.5f}; p1 = {learner.P1:.5f}")
