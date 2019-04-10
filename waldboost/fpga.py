"""
Support for FPGA stuff - like feature banks, quantization etc.
"""


import numpy as np
from scipy.ndimage import convolve1d


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


class Quantizer:
    def __init__(self, range=(-1,1), n_bits=8, type="right"):
        a,b = range
        self.bins = np.linspace(a, b, (2**n_bits)+1)[:-1]
        self.type = type
        self.offset = np.digitize(0, self.bins, self.type)

    def __call__(self, x):
        b = np.maximum(np.digitize(x, self.bins, self.type)-1,0)
        q = self.bins[b]
        return q, b

    @property
    def bin_edges(self):
        return self.bins
