import numpy as np


class LaserData(object):
    DEFAULT_CONFIG = {
        'spotsize': 30.0, 'scantime': 0.25, 'speed': 120.0,
        'gradient': 1.0, 'intercept': 0.0}

    def __init__(self, data=None, isotope="", config=None,
                 source=""):
        self.data = data
        self.isotope = isotope
        self.config = LaserData.DEFAULT_CONFIG if config is None else config
        self.source = source

    def calibrated(self):
        return (self.data - self.config['intercept']) / self.config['gradient']

    def pixelsize(self):
        return (self.config['speed'] * self.config['scantime'],
                self.config['spotsize'])

    def aspect(self):
        return 1.0 / ((self.config['speed'] * self.config['scantime'] /
                       self.config['spotsize']))

    def extent(self):
        x = self.data.shape[1] * self.pixelsize()[0]
        y = self.data.shape[0] * self.pixelsize()[1]
        return (0, x, 0, y)


class KrissKrossData(LaserData):
    def __init__(self, data=None, isotope="", config=None, source=""):
        super().__init__(self, data=data, isotpe=isotope,
                         config=config, source=source)

    def flatten(self):
        return np.mean(self.data, axis=2)

    def calibrated(self, flat=False):
        return np.mean(super().calibrated(), axis=2) if flat \
               else super().calibrated()
