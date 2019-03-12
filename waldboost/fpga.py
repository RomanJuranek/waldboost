"""
Support for FPGA stuff - like feature banks etc.
"""


def bank_pattern(shape, n_banks=4):
    pass np.arange(np.prod(shape)).reshape(shape) % n_banks


def bank_id(idx, n_banks=4):
    return idx % n_banks
