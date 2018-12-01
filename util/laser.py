import numpy as np
from collections import defaultdict


class LaserData(object):
    DEFAULT_CONFIG = {
        'spotsize': 30.0, 'speed': 120.0, 'scantime': 0.25,
        'gradient': defaultdict(1.0), 'intercept': defaultdict(0.0)}

    def __init__(self, data=None, config=None,
                 source=""):
        self.data = data
        self.config = LaserData.DEFAULT_CONFIG if config is None else config
        self.source = source

    def isotopes(self):
        return self.data.dtype.names

    def calibrated(self, isotope=None):
        if isotope is None:
            data = np.empty(self.data.shape, dtype=self.data.dtype)
            for name in self.data.names:
                data[name] = ((self.data[name]
                              - self.config['intercepts'][name])
                              / self.config['gradients'][name])
        else:
            data = ((self.data[isotope] - self.config['intercepts'][isotope])
                    / self.config['gradients'][isotope])
        return data

    def pixelsize(self):
        return (self.config['speed'] * self.config['scantime'],
                self.config['spotsize'])

    def aspect(self):
        return self.config['spotsize'] / \
               (self.config['speed'] * self.config['scantime'])

    def extent(self):
        # Image data is stored [rows][cols]
        x = self.data.shape[1] * self.pixelsize()[0]
        y = self.data.shape[0] * self.pixelsize()[1]
        return (0, x, 0, y)
