import numba as nb
import numpy as np

from waldboost.channels import _grad_x, _grad_y


@nb.njit(["i4[:,:,:](u1[:,:])"], nogil=True)
def _grad_hist_4_int(arr):
    """
    8bit input -> 4 channel 4bit
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
    return y


def grad_hist_4_u1(image):
    """
    scale of abs(y) is +-1024
    y/4 - scale to original domain +-256
    To scale the full range to +-256 -> y/4 - small gradients lost
    This is prolly ok
    """
    y = _grad_hist_4_int(image)
    np.fmin(np.abs(y)>>2, np.int32(255), out=y)
    return y


@nb.njit(["i4[:,:,:](u1[:,:])"], nogil=True, cache=True)
def _grad_mag_int(arr):
    dx = np.abs(_grad_x(arr))
    dy = np.abs(_grad_y(arr))
    dst_shape = (arr.shape[0], arr.shape[1], 1)
    y = np.empty(dst_shape, np.int32)
    y[...,0] = np.maximum(dx, dy) >> 2
    np.fmin(y, np.int32(255))
    return y


def grad_mag_u1(image):
    return _grad_mag_int(image)
