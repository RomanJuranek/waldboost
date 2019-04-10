import logging
import numpy as np
from skimage.measure import block_reduce
from skimage.transform import resize
from scipy.ndimage import convolve1d


logger = logging.getLogger(__name__)


def triangle_kernel(n):
    H = (np.r_[:n+1,n-1:-1:-1]+1).astype("f")
    return H / H.sum()


def gradients(image):
    D = np.array( [1,0,-1], "f")
    gy = convolve1d(image, D, axis=0)
    gx = convolve1d(image, D, axis=1)
    return gx, gy


def separable_convolve(image, kernel):
    output = convolve1d(image, kernel, axis=0)
    convolve1d(output, kernel, axis=1, output=output)
    return output


def grad_mag(image, norm=None, eps=1e-3):
    gx, gy = gradients(image)
    mag = np.sqrt(gx**2 + gy**2)
    if norm is not None and norm > 1:
        H = triangle_kernel(norm)
        norm = separable_convolve(mag, H)
        mag /= norm + eps
    return mag[...,None]


def grad_hist(image, n_bins=6, full=False):
    gx, gy = gradients(image)
    max_theta = 2*np.pi if full else np.pi
    theta = np.linspace(0, max_theta, n_bins+1)
    cs = np.cos(theta[:-1])
    sn = np.sin(theta[:-1])
    u,v = gx.shape
    chns = np.empty((u,v,n_bins), gx.dtype)
    for i,(c,s) in enumerate(zip(cs,sn)):
        chns[...,i] = gx*c - gy*s;
    if full:
        return np.fmax(chns, 0)
    else:
        return np.abs(chns)


def channel_pyramid(image, opts):
    pyr_opts = opts["pyramid"]
    shrink = pyr_opts["shrink"]
    n_per_oct = pyr_opts["n_per_oct"]
    smooth = pyr_opts["smooth"]
    channels = pyr_opts["channels"]
    target_dtype = pyr_opts["target_dtype"]

    assert shrink in [1,2,3,4], "Shrink factor must be integer 1 <= shrink <= 4"

    base_image = image
    m,n,n_channels =  opts["shape"]

    factor = 2**(-1/n_per_oct)

    while True:
        h,w = base_image.shape
        for i in range(n_per_oct):
            s = factor ** i
            nw, nh = int((w*s)/shrink)*shrink, int((h*s)/shrink)*shrink
            if nh < shrink*m or nw < shrink*n:
                return

            real_scale = nw / image.shape[1]
            im = (255*resize(base_image, (nh, nw))).astype("u1")

            if channels:
                chns = [ func(im, *pfunc) for func,pfunc in channels ]
            else:
                chns = [ im[...,None] ]

            chns = np.concatenate(chns, axis=-1)
            if shrink > 1:
                chns = block_reduce(chns, (shrink,shrink,1), np.mean).astype(target_dtype)

            if smooth > 0:
                H = triangle_kernel(smooth)
                chns = separable_convolve(chns, H)

            yield np.atleast_3d(chns), real_scale/shrink

        base_image = block_reduce(base_image, (2,2), np.mean).astype(image.dtype)
