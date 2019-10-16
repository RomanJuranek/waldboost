import numba as nb
import numpy as np


@nb.stencil(neighborhood=((-1,1),(-1,1)))
def _grad_x(arr):
    """ Convolution with horizontal derivative kernel
    H = [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]
    """
    dx = -(arr[-1,-1] + 2*arr[0,-1] + arr[1,-1]) + \
           arr[-1, 1] + 2*arr[0, 1] + arr[1, 1]
    return dx


@nb.stencil(neighborhood=((-1,1),(-1,1)))
def _grad_y(arr):
    """ Convolution with vertical derivative kernel
    H = [[-1,-2,-1],
         [ 0, 0, 0],
         [ 1, 2, 1]]
    """
    dy = -(arr[-1,-1] + 2*arr[-1,0] + arr[-1,1]) + \
           arr[ 1,-1] + 2*arr[ 1,0] + arr[ 1,1]
    return dy


@nb.njit(nogil=True)
def _grad_hist_4_u1(arr):
    """
    8bit input -> 4 channel 32bit (but clamped to 8 bits)

    scale of abs(y) is +-1024
    y/4 - scale to original domain +-256
    To scale the full range to +-256 -> y/4 - small gradients lost
    This is prolly ok
    """
    dst_shape = (arr.shape[0], arr.shape[1], 4)
    dx = np.empty(arr.shape, np.int32)
    dy = np.empty(arr.shape, np.int32)
    dx[:] = _grad_x(arr)
    dy[:] = _grad_y(arr)
    y = np.empty(dst_shape, np.int32)
    y[...,0] = dx
    y[...,1] = 0.5 * dx - 0.5 * dy
    y[...,2] = dy
    y[...,3] = 0.5 * dx + 0.5 * dy
    return np.fmin(np.abs(y)//4, 255).astype(np.uint8)


def grad_hist_4_u1(image):
    return _grad_hist_4_u1(image)


@nb.njit(nogil=True)
def _grad_mag_u1(arr):
    dx = np.abs(_grad_x(arr))
    dy = np.abs(_grad_y(arr))
    dst_shape = (arr.shape[0], arr.shape[1], 1)
    y = np.empty(dst_shape, np.int32)
    y[...,0] = np.maximum(dx, dy)
    return np.fmin(y//4, 255).astype(np.uint8)


def grad_mag_u1(image):
    return _grad_mag_u1(image)