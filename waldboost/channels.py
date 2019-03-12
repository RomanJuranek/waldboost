import logging
import numpy as np
from skimage.measure import block_reduce
from scipy.ndimage import convolve
import cv2


logger = logging.getLogger(__name__)


deriv_H = np.array((-1,0,1), np.float) / 2


def triangle_kernel(n):
    H = (np.r_[:n+1,n-1:-1:-1]+1).astype("f")
    return H / H.sum()


def gradients(image):
    gx = cv2.filter2D(image, cv2.CV_32F, deriv_H[None,:])
    gy = cv2.filter2D(image, cv2.CV_32F, deriv_H[:,None])
    return gx, gy


def grad_mag(image, norm=None, eps=1e-3):
    gx, gy = gradients(image)
    mag = cv2.magnitude(gx, gy)
    if norm is not None:
        H = triangle_kernel(norm)
        norm = cv2.sepFilter2D(mag, cv2.CV_32F, H, H)
        mag /= norm + eps
    return mag[...,None]


def grad_hist(image, n_bins=6, full=False):
    gx, gy = gradients(image)
    max_theta = 2*np.pi if full else np.pi
    theta = np.linspace(0, max_theta, n_bins+1)
    cs = np.cos(theta[:-1])
    sn = np.sin(theta[:-1])
    u,v = dx.shape
    chns = np.empty((u,v,n_bins), dx.dtype)
    for i,(c,s) in enumerate(zip(cs,sn)):
        chns[...,i] = dx*c - dy*s;
    if full:
        return np.fmax(chns, 0)
    else:
        return np.abs(chns)


def channel_pyramid(image, opts):
    pyr_opts = opts["pyramid"]
    shrink = pyr_opts["shrink"]
    n_per_oct = pyr_opts["n_per_oct"]
    smooth = pyr_opts["smooth"]
    factor = 2**(-1/n_per_oct)

    assert shrink in [1,2,3,4], "Shrink factor can be only integer 1 to 4"

    base_image = image
    m,n,n_channels =  opts["shape"]

    while True:
        h,w = base_image.shape
        for i in range(n_per_oct):
            s = factor ** i
            nw, nh = int((w*s)/shrink)*shrink, int((h*s)/shrink)*shrink
            if nh < shrink*m or nw < shrink*n:
                return

            real_scale = nw / image.shape[1]
            im = cv2.resize(base_image, (nw, nh), cv2.INTER_LINEAR)
            im = im.astype(np.float32) / 256

            chns = grad_mag(im, norm=5)
            chns = block_reduce(chns, (2,2,1), np.max)
            if smooth > 0:
                H = np.dot(triangle_kernel(smooth)[:,None], triangle_kernel(smooth)[None,:])[...,None]
                chns = convolve(chns, H)

            yield chns, real_scale/shrink

        #print("downscale")
        base_image = cv2.resize(base_image, (w//2, h//2))
