"""
Support for FPGA related stuff

Classes and function provided in this package more or less
replace their default counterparts, e.g. waldboost.fpga.grad_hist_4_uint8 can
be used instead waldboost.channels.grad_hist.
"""

from .channels import grad_hist_4_u1, grad_mag_u1
from .training import DTree, train
