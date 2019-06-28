"""
Support for FPGA stuff - like feature banks, quantization etc.
"""

import logging
import numpy as np
from scipy.ndimage import convolve1d
from .training import DTree, as_features
from itertools import count
from queue import Queue


def bank_pattern(shape, block_shape=(2,2)):
    assert len(shape) in [2,3], "Shape must be (H,W) or (H,W,C)"
    if len(shape) == 2:
        shape += (1,)
    b = np.arange(np.prod(block_shape)).reshape(block_shape)
    n = np.ceil(np.array(shape[:2]) / block_shape)
    banks = np.tile(b, n.astype("i").tolist())
    u,v,c = shape
    banks = np.atleast_3d(np.dstack([banks]*c))
    return banks[:u,:v,...]


def grad_hist_4(image):
    """
    Integer version of grad_hist(image, n_bins=4, full=False)
    Input image must be uint8
    Output image is always int16
    """
    assert image.dtype == np.uint8

    H = np.array( [1,2,1], "i2")
    D = np.array( [1,0,-1], "i2")

    im = convolve1d(image.astype("i2"),H,axis=0)
    convolve1d(im,H,axis=1,output=im)
    im = np.right_shift(im, 4)

    gy = convolve1d(im, D, axis=0) >> 1
    gx = convolve1d(im, D, axis=1) >> 1

    chns = np.empty(im.shape + (4,), "i2")

    theta = np.linspace(0, np.pi, 5)
    cs = np.cos(theta[:-1])
    sn = np.sin(theta[:-1])

    for i,(c,s) in enumerate(zip(cs,sn)):
        chns[...,i] = gx*c - gy*s;

    return np.abs(chns)


# def fit_feature(x,y,w):
#     f = np.random.randint(x.shape[1])
#     t = np.median(x[:,f])
#     return f, t


def gini(w0, w1):
    return 1 - (w0**2 + w1**2)


def entropy(w0, w1):
    return -((w0*np.log2(w0)) + (w1*np.log2(w1)))


def fit_threshold(v0, v1, w0, w1, vmin, vmax):
    # print("values:", v0.shape, v1.shape)
    # print("weights:", w0.shape, w1.shape)
    # print("range:", vmin, vmax)
    bins = np.r_[vmin-1:vmax+1]
    h0,_ = np.histogram(v0, bins, weights=w0)
    h1,_ = np.histogram(v1, bins, weights=w1)
    h0 = np.cumsum(h0)
    h1 = np.cumsum(h1)
    w = h0 + h1
    h0_max = h0[-1]
    h1_max = h1[-1]
    h_max = h0_max + h1_max
    metric = entropy(h0_max,h1_max) - (w/h_max)*entropy(h0,h1) - ((h_max-w)/h_max)*entropy(h0_max-h0,h1_max-h1)
    k = np.argmin(metric)
    return bins[k+1], metric[k]


def fit_feature(x, y, w):
    n_samples, n_features = x.shape
    t_min,t_max = x.min(axis=0), x.max(axis=0)
    x0,w0 = x[y==0].T,w[y==0]
    x1,w1 = x[y==1].T,w[y==1]
    threshold, metric = zip(*[fit_threshold(v0,v1,w0,w1,t0,t1) for v0,v1,t0,t1 in zip(x0,x1,t_min,t_max)])
    k = np.argmax(metric)
    return k, threshold[k], metric[k]


class FPGA_DTree:
    @staticmethod
    def fit(X0, W0, X1, W1, max_depth=2):
        shape = X0.shape[1:]

        X = np.concatenate( [as_features(X0), as_features(X1)] )
        Y = np.array([0]*X0.shape[0] + [1]*X1.shape[0])
        W = np.concatenate( [W0, W1] )

        n_samples = W.size
        W[Y==0] /= W[Y==0].sum() * 2
        W[Y==1] /= W[Y==1].sum() * 2
        #W /= W.sum()
        print(W.sum(), W[Y==0].sum(), W[Y==1].sum())

        node_id = count()
        sample_queue = Queue()
        sample_queue.put( (np.arange(n_samples), 0, next(node_id)) )
        nodes = dict() # Intermediate representation of tree
        while not sample_queue.empty():
            self_samples, depth, self_index = sample_queue.get()

            # Create either leaf or split node
            create_leaf = (depth == max_depth) or (self_samples.size < 10)
            if create_leaf:
                nodes[self_index] = {"samples": self_samples,
                                     "feature": -1, "threshold": -1,
                                     "left": -1, "right": -1}
                # logging.info(f"Leaf node {nodes[self_index]}")
            else:
                feature, threshold, _ = fit_feature(X[self_samples], Y[self_samples], W[self_samples])
                # split data
                bin = X[self_samples, feature] <= threshold
                # schedule l/r nodes
                left_index = next(node_id)
                sample_queue.put( (self_samples[ bin], depth+1, left_index) )
                right_index = next(node_id)
                sample_queue.put( (self_samples[~bin], depth+1, right_index) )
                # create node
                nodes[self_index] = {"samples": self_samples,
                                     "feature": feature, "threshold": threshold,
                                     "left": left_index, "right": right_index}

        n_nodes = len(nodes)
        feature = [None] * n_nodes
        threshold = np.empty(n_nodes)
        left = np.empty(n_nodes, "i")
        right = np.empty(n_nodes, "i")
        pred = np.empty(n_nodes, "f")

        for node_idx, node_data in nodes.items():
            f = node_data["feature"]
            feature[node_idx] = np.unravel_index(f, shape) if f>=0 else None
            threshold[node_idx] = node_data["threshold"]
            left[node_idx] = node_data["left"]
            right[node_idx] = node_data["right"]
            idx = node_data["samples"]
            y, w = Y[idx], W[idx]
            w0 = w[y==0].sum() + 1e-4
            w1 = w[y==1].sum() + 1e-4
            pred[node_idx] = np.round(32 * 0.5 * np.log(w1/w0)) / 32

        #print(feature, threshold, left, right, pred)

        return DTree(feature, threshold, left, right, pred)


channel_opts = {
    "shrink": 2,
    "n_per_oct": 8,
    "smooth": 0,
    "target_dtype": np.int16,
    "channels": [ grad_hist_4 ],
    }
