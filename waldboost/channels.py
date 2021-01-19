import logging

import numba as nb
import numpy as np
from scipy.ndimage import convolve1d
from skimage.transform import resize

logger = logging.getLogger(__name__)


def triangle_kernel(n):
    H = (np.r_[:n+1,n-1:-1:-1]+1).astype("f")
    return H / H.sum()


def gradients(image):
    H = np.array([1,2,1], "f4")
    D = np.array([-1,0,1], "f4")
    gy = convolve1d(convolve1d(image,H,axis=1), D, axis=0)
    gx = convolve1d(convolve1d(image,H,axis=0), D, axis=1)
    return gx, gy


def separable_convolve(image, k0, k1=None):
    output = convolve1d(image, k0, axis=0)
    convolve1d(output, k1 or k0, axis=1, output=output)
    return output


def grad_mag(image, norm=5, eps=1e-3):
    gx, gy = gradients(image.astype("f"))
    mag = np.sqrt(gx**2 + gy**2)
    if norm is not None and norm > 1:
        H = triangle_kernel(norm)
        norm = separable_convolve(mag, H)
        mag /= norm + eps
    return mag[...,None]


def grad_hist(image, n_bins=4, full=False, bias=0):
    image = image.astype("f")
    gx, gy = gradients(image)
    max_theta = 2*np.pi if full else np.pi
    theta = np.linspace(0, max_theta, n_bins+1)
    cs = np.cos(theta[:-1])
    sn = np.sin(theta[:-1])
    u,v = gx.shape
    chns = np.empty((u,v,n_bins), gx.dtype)
    for i,(c,s) in enumerate(zip(cs,sn)):
        chns[...,i] = gx*c - gy*s
    chns_value = np.fmax(np.abs(chns)-bias, 0)
    return np.sign(chns) * chns_value if full else chns_value


@nb.njit(nogil=True)
def avg_pool_2(arr):
    u = arr.shape[0]
    v = arr.shape[1]
    u_lim = u - (u%2)
    v_lim = v - (v%2)
    return ((arr[0:u_lim:2,0:v_lim:2,...] +
            arr[1:u_lim:2,0:v_lim:2,...] +
            arr[0:u_lim:2,1:v_lim:2,...] +
            arr[1:u_lim:2,1:v_lim:2,...]) / 4).astype(arr.dtype)


@nb.njit(nogil=True)
def max_pool_2(arr):
    u = arr.shape[0]
    v = arr.shape[1]
    u_lim = u - (u%2)
    v_lim = v - (v%2)
    m0 = np.fmax(arr[0:u_lim:2,0:v_lim:2,...], arr[1:u_lim:2,0:v_lim:2,...])
    m1 = np.fmax(arr[0:u_lim:2,1:v_lim:2,...], arr[1:u_lim:2,1:v_lim:2,...])
    return np.fmax(m0, m1)


@nb.stencil(neighborhood=((-1,1),(-1,1),(0,0)) )
def _smooth(arr):
    v =   arr[-1,-1,0] + 2*arr[-1,0,0] +   arr[-1,1,0] + \
        2*arr[ 0,-1,0] + 4*arr[ 0,0,0] + 2*arr[ 0,1,0] + \
          arr[ 1,-1,0] + 2*arr[ 1,0,0] +   arr[ 1,1,0]
    return v


@nb.njit(nogil=True)
def smooth_image_3d(arr):
    smoothed = np.empty_like(arr)
    smoothed[:] = _smooth(arr) / 16
    return smoothed


def _image_octaves(image):
    """ Generate downsampled images """
    base_image = image.copy()
    while True:
        h,w = base_image.shape[:2]
        if w < 8 or h < 8:
            break
        yield base_image
        base_image = avg_pool_2(base_image)


def _validate_image(image:np.ndarray):
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be numpy array")
    if image.ndim != 2:
        raise ValueError("Image must have 2 dimensions")


def channel_pyramid(image, channel_opts):
    """ Generate image pyramid """

    _validate_image(image)

    shrink = channel_opts["shrink"]
    n_per_oct = channel_opts["n_per_oct"]
    smooth = channel_opts["smooth"]
    channels = channel_opts["channels"]
    assert shrink in [1,2], "Shrink factor must be integer 1 <= shrink <= 2"

    dtype = image.dtype

    factor = 2**(-1/n_per_oct)
    for base_image in _image_octaves(image):
        h,w,*_ = base_image.shape
        for i in range(n_per_oct):
            # Get the size for level i
            s = factor ** i
            nw, nh = int((w*s)/shrink)*shrink, int((h*s)/shrink)*shrink
            real_scale = nw / image.shape[1]
            im = resize(base_image, (nh, nw), preserve_range=True, order=1).astype(dtype)
            
            #print(im.shape, im.dtype, im.max())

            chns = channels(im)

            if shrink == 2:
                chns = avg_pool_2(chns)

            if smooth == 1:
                chns = smooth_image_3d(chns)

            #print("CHANNELS: ", chns.shape, chns.dtype, chns.max())

            yield np.atleast_3d(chns), real_scale/shrink
