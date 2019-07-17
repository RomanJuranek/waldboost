"""
Channel computation accelerated with numba
"""


import numba as nb
import numpy as np


@nb.stencil(neighborhood=((-1,1),(-1,1)))
def _grad_x(arr):
    dx = -(arr[-1,-1] + 2*arr[0,-1] + arr[1,-1]) + \
           arr[-1, 1] + 2*arr[0, 1] + arr[1, 1]
    return dx


@nb.stencil(neighborhood=((-1,1),(-1,1)))
def _grad_y(arr):
    dy = -(arr[-1,-1] + 2*arr[-1,0] + arr[-1,1]) + \
           arr[ 1,-1] + 2*arr[ 1,0] + arr[ 1,1]
    return dy

_signatures = [
    "i4[:,:,:](u1[:,:],i4)",
]

@nb.njit(_signatures, nogil=True)
def _grad_hist_4(arr, bias):
    dst_shape = (arr.shape[0], arr.shape[1], 4)
    dx = np.empty(arr.shape, np.int32)
    dy = np.empty(arr.shape, np.int32)
    dx[:] = _grad_x(arr)
    dy[:] = _grad_y(arr)
    y = np.empty(dst_shape, nb.int32)
    y[...,0] = dx
    y[...,1] = 0.7 * dx - 0.7 * dy
    y[...,2] = dy
    y[...,3] = 0.7 * dx + 0.7 * dy
    return np.fmax(np.abs(y)-bias, np.int32(0))


def grad_hist_4(arr, bias=4):
    return _grad_hist_4(arr, bias)