from itertools import count

import numpy as np


def _bank_pattern(shape, block_shape):
    assert len(shape) in [2,3], "Shape must be (H,W) or (H,W,C)"
    if len(shape) == 2:
        shape += (1,)
    b = np.arange(np.prod(block_shape)).reshape(block_shape)
    n = np.ceil(np.array(shape[:2]) / block_shape)
    banks = np.tile(b, n.astype("i").tolist())
    u,v,c = shape
    banks = np.atleast_3d(np.dstack([banks]*c))
    return banks[:u,:v,...]


class PixelBanks:
    def __init__(self, shape, block_shape):
        self.pattern = _bank_pattern(shape, block_shape)

    def bank_pixels(self, bank_ids):
        return np.concatenate( [np.flatnonzero(self.pattern == b) for b in bank_ids] )


class BankScheduler:
    def __init__(self, n_banks=8):
        self.n_banks = n_banks
        self.bank_counter = count()

    def schedule(self, max_depth=2):
        return [ [next(self.bank_counter) % self.n_banks] for _ in range(max_depth)]
