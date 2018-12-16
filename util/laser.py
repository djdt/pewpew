import numpy as np


class LaserData(object):
    DEFAULT_CALIBRATION = {"gradients": {}, "intercepts": {}, "units": {}}
    DEFAULT_CONFIG = {
        "spotsize": 30.0,
        "speed": 120.0,
        "scantime": 0.25,
        "trim": [0, 0],
    }

    def __init__(self, data=None, config=None, calibration=None, name="", source=""):
        self.data = (
            np.array([[0]], dtype=[("NONE", np.float64)]) if data is None else data
        )
        self.config = LaserData.DEFAULT_CONFIG if config is None else config
        self.calibration = (
            LaserData.DEFAULT_CALIBRATION if calibration is None else calibration
        )
        self.name = name
        self.source = source

    def isotopes(self):
        return self.data.dtype.names

    def get(self, isotope=None, calibrated=False, trimmed=False):
        # Calibration
        if isotope is None:
            data = self.data
            if calibrated:
                for name in data.dtype.names:
                    intercept = self.calibration["intercepts"].get(name, 0.0)
                    gradient = self.calibration["gradients"].get(name, 1.0)
                    data[name] = (data[name] - intercept) / gradient
        else:
            data = self.data[isotope]
            if calibrated:
                intercept = self.calibration["intercepts"].get(isotope, 0.0)
                gradient = self.calibration["gradients"].get(isotope, 1.0)
                data = (data - intercept) / gradient
        # Trimming
        if trimmed:
            trim = self.config["trim"]
            if trim[1] > 0:
                data = data[:, trim[0] : -trim[1]]
            elif trim[0] > 0:
                data = data[:, trim[0] :]

        return data

    def pixelsize(self):
        return (self.config["speed"] * self.config["scantime"], self.config["spotsize"])

    def aspect(self):
        return self.config["spotsize"] / (
            self.config["speed"] * self.config["scantime"]
        )

    def extent(self, trimmed=False):
        # Image data is stored [rows][cols]
        x_shape = self.data.shape[1]
        if trimmed:
            trim = self.config["trim"]
            x_shape -= trim[0] + trim[1]
        x = x_shape * self.pixelsize()[0]
        y = self.data.shape[0] * self.pixelsize()[1]
        return [0, x, 0, y]

    def setTrim(self, trim, unit="rows"):
        """Set the trim value using the provided unit.
        Valid units are 'rows', 'μm' and 's'."""
        if unit == "μm":
            width = self.pixelsize()[0]
            trim = [trim[0] / width, trim[1] / width]
        elif unit == "s":
            width = self.config["scantime"]
            trim = [trim[0] / width, trim[1] / width]
        self.config["trim"] = [int(trim[0]), int(trim[1])]

    def trimAs(self, unit):
        """Returns the trim in given unit.
        Valid units are 'rows', 'μm' and 's'."""
        trim = self.config["trim"]
        if unit == "μm":
            width = self.pixelsize()[0]
            trim = [trim[0] * width, trim[1] * width]
        elif unit == "s":
            width = self.config["scantime"]
            trim = [trim[0] * width, trim[1] * width]
        return trim

    def layers(self):
        return 1
