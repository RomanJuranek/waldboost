"""
Support for FPGA related stuff

Classes and function provided in this package more or less
replace their default counterparts.

waldboost.channels.grad_hist  ->  waldboost.fpga.grad_hist_4_u1
waldboost.DTree  ->  waldboost.fpga.DTree
waldboost.train  ->  waldboost.fpga.train
"""

from .channels import grad_hist_4_u1, grad_mag_u1
from .training import DTree, train
