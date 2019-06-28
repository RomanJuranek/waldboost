def takemin(iterable, objective):
    """
    Return an element and its position form an iterable which evaluates
    an objective function to a minimum value among other elements from the
    iterable. This is equivalent to argmin and min functions when the objective
    is simply the value of element. But it can be used in more general settings.

    idx, element = takemin(iterable, objective)

    Example:
    X = np.random.randn(1000)
    idx, value = takemin(X, objective=lambda x: np.abs(x-1))

    This will return index and value of element of X closest to 1.
    """
    take_el = None
    best = None
    best_idx = None
    for idx, el in enumerate(iterable):
        m = objective(el)
        if best is None or m < best:
            best = m
            take_el = el
            best_idx = idx
    return best_idx, take_el


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


def draw_detections(img, model, v=None, nms=False, min_score=None, v0=None, v1=None):
    import numpy as np
    import matplotlib as mpl
    from matplotlib import cm
    import bbx
    from .verification import detect_and_verify
    import cv2
    #cv2.namedWindow("x", cv2.WINDOW_NORMAL)

    if not v:
        bbs,scores = model.detect(img)
    else:
        bbs,scores = detect_and_verify(img, model, v)

    if v0 is None:
        v0, v1 = scores.min(), scores.max()

    if min_score is not None:
        mask = scores > min_score
        bbs,scores = bbs[mask],scores[mask]

    if nms:
        bbs,scores = bbx.nms(bbs, scores, min_group=4, min_overlap=0.3)

    N = mpl.colors.Normalize(vmin=v0, vmax=v1)
    colors = cm.plasma;
    I = cv2.cvtColor(img[...,0], cv2.COLOR_GRAY2BGR)
    order = np.argsort(scores)
    for (x,y,w,h),score in zip(bbs[order].astype("i"), scores[order]):
        clr = (255*np.array(colors(N(score)))[2::-1]).astype("u1")
        clr = tuple(map(int, clr))
        cv2.rectangle(I, (x,y), (x+w,y+h), clr, thickness=1)

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
