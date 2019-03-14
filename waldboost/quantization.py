import numpy as np


class Quantizer:
    def __init__(self, range=(-1,1), n_bits=8, type="right"):
        a,b = range
        self.bins = np.linspace(a, b, (2**n_bits)+1)[:-1]
        self.type = type
        self.offset = np.digitize(0, self.bins, self.type)

    def __call__(self, x):
        return np.digitize(x, self.bins, self.type) - self.offset

    @property
    def bin_edges(self):
        return self.bins
