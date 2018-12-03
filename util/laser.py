import numpy as np


class LaserData(object):
    DEFAULT_CALIBRATION = {'gradients': {}, 'intercepts': {}, 'units': {}}
    DEFAULT_CONFIG = {
        'spotsize': 30.0,
        'speed': 120.0,
        'scantime': 0.25,
    }

    def __init__(self, data=None, config=None, calibration=None, source=""):
        self.data = data
        self.config = LaserData.DEFAULT_CONFIG if config is None else config
        self.calibration = LaserData.DEFAULT_CALIBRATION \
            if calibration is None else calibration
        self.source = source

    def isotopes(self):
        return self.data.dtype.names

    def calibrated(self, isotope=None):
        if isotope is None:
            data = np.empty(self.data.shape, dtype=self.data.dtype)
            for name in self.data.dtype.names:
                data[name] = ((self.data[name]
                               - self.calibration['intercepts'].get(name, 0.0))
                              / self.calibration['gradients'].get(name, 0.0))
        else:
            data = ((self.data[isotope] - self.calibrated['intercepts'].get(
                isotope, 0.0)) / self.calibration['gradients'].get(
                    isotope, 1.0))
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

    def layers(self):
        return 1
