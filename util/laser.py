import numpy as np


class LaserData(object):
    DEFAULT_CALIBRATION = {'gradients': {}, 'intercepts': {}, 'units': {}}
    DEFAULT_CONFIG = {
        'spotsize': 30.0,
        'speed': 120.0,
        'scantime': 0.25,
        'trim': [0, 0],
    }

    def __init__(self,
                 data=None,
                 config=None,
                 calibration=None,
                 trim=[0, 0],
                 name="",
                 source=""):
        self.data = data
        self.config = LaserData.DEFAULT_CONFIG if config is None else config
        self.calibration = LaserData.DEFAULT_CALIBRATION \
            if calibration is None else calibration
        self.trim = trim
        self.name = name
        self.source = source

    def calibrated(self, isotope=None):
        if isotope is None:
            data = np.empty(self.data.shape, dtype=self.data.dtype)
            for name in self.data.dtype.names:
                data[name] = ((self.data[name] -
                               self.calibration['intercepts'].get(name, 0.0)) /
                              self.calibration['gradients'].get(name, 0.0))
        else:
            data = ((self.data[isotope] - self.calibration['intercepts'].get(
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

    def setTrim(self, trim, unit='rows'):
        """Set the trim value using the provided unit.
        Valid units are 'rows', 'μm' and 's'."""
        if unit == "μm":
            width = self.pixelsize()[0]
            self.trim = [trim[0] / width, trim[1] / width]
        elif unit == "s":
            width = self.config['scantime']
            trim = [trim[0] / width, trim[1] / width]
        self.trim = [int(trim[0]), int(trim[1])]

    def trimAs(self, unit):
        """Returns the trim in given unit.
        Valid units are 'rows', 'μm' and 's'."""
        trim = self.trim
        if unit == "μm":
            width = self.pixelsize()[0]
            trim = [trim[0] * width, trim[1] * width]
        elif unit == "s":
            width = self.config['scantime']
            trim = [trim[0] * width, trim[1] * width]
        return trim

    def layers(self):
        return 1
