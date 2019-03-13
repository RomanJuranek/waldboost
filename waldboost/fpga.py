"""
Support for FPGA stuff - like feature banks etc.
"""


import numpy as np
from scipy.ndimage import convolve1d


# def bank_pattern(shape, n_banks=4):
#     pass np.arange(np.prod(shape)).reshape(shape) % n_banks
#
#
# def bank_id(idx, n_banks=4):
#     return idx % n_banks


def grad_hist_4(image):
    """
    Integer version of grad_hist(image, n_bins=4, full=False)
    """

    assert image.dtype == np.uint8

    H = np.array( [1,2,1], "i2")
    D = np.array( [1,0,-1], "i2")

    im = convolve1d(convolve1d(image.astype("i2"),H,axis=0),H,axis=1) >> 4  # back to 8 bits
    gy = convolve1d(im, D, axis=0) >> 1
    gx = convolve1d(im, D, axis=1) >> 1

    chns = np.empty(im.shape + (4,), "i2")

    theta = np.linspace(0, np.pi, 5)
    cs = np.cos(theta[:-1])
    sn = np.sin(theta[:-1])

    for i,(c,s) in enumerate(zip(cs,sn)):
        chns[...,i] = gx*c - gy*s;

    return np.abs(chns)
