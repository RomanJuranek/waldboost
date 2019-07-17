import logging
import numpy as np
import numba as nb
from skimage.measure import block_reduce
from skimage.transform import resize
from scipy.ndimage import convolve1d
import cv2

logger = logging.getLogger(__name__)


def triangle_kernel(n):
    H = (np.r_[:n+1,n-1:-1:-1]+1).astype("f")
    return H / H.sum()


def gradients(image):
    D = np.array( [1,0,-1], "f")
    gy = convolve1d(image, D, axis=0)
    gx = convolve1d(image, D, axis=1)
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


def grad_hist(image, n_bins=6, full=False, bias=0):
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
    if full:
        return np.sign(chns) * chns_value
    else:
        return chns_value


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


@nb.stencil(neighborhood=((-1,1),(-1,1)))
def _smooth(arr):
    v =   arr[-1,-1] + 2*arr[-1,0] +   arr[-1,1] + \
        2*arr[ 0,-1] + 4*arr[ 0,0] + 2*arr[ 0,1] + \
          arr[ 1,-1] + 2*arr[ 1,0] +   arr[ 1,1]
    return v / 16


@nb.njit(nogil=True)
def smooth_image_3d(arr):
    smoothed = np.empty_like(arr)
    for k in range(arr.shape[2]):
        smoothed[...,k] = _smooth(arr[...,k])
    return smoothed


def _image_octaves(image, min_size=(16,16)):
    base_image = image.copy()
    while True:
        yield base_image
        h,w = base_image.shape[:2]
        if ((w//2,h//2) < min_size):
            break
        base_image = avg_pool_2(base_image)


def channel_pyramid(image, channel_opts):
    shrink = channel_opts["shrink"]
    n_per_oct = channel_opts["n_per_oct"]
    smooth = channel_opts["smooth"]
    channels = channel_opts["channels"]
    # target_dtype = channel_opts["target_dtype"]
    assert shrink in [1,2], "Shrink factor must be integer 1 <= shrink <= 2"

    factor = 2**(-1/n_per_oct)
    for base_image in _image_octaves(image):
        h,w,*_ = base_image.shape
        for i in range(n_per_oct):
            s = factor ** i
            nw, nh = int((w*s)/shrink)*shrink, int((h*s)/shrink)*shrink
            real_scale = nw / image.shape[1]

            im = cv2.resize(base_image, (nw, nh), cv2.INTER_LINEAR)[...,None]

            if channels:
                chns = [func(im[...,0]) for func in channels]
                chns.append(im[...,1:])
                chns = np.concatenate(chns, axis=-1)
            else:
                chns = im

            if shrink == 2:
                chns = avg_pool_2(chns)

            if smooth == 1:
                chns = smooth_image_3d(chns)

            yield np.atleast_3d(chns), real_scale/shrink
