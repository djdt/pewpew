from typing import List, Optional, Tuple
import numpy as np


class LaserData(object):
    DEFAULT_CALIBRATION = {"gradient": 1.0, "intercept": 0.0, "unit": None}
    DEFAULT_CONFIG = {
        "spotsize": 30.0,
        "speed": 120.0,
        "scantime": 0.25,
        "trim": (0, 0),
    }

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        config: Optional[dict] = None,
        calibration: Optional[dict] = None,
        name: str = "",
        source: str = "",
    ):
        self.data = (
            np.array([[0]], dtype=[("none", np.float64)]) if data is None else data
        )
        self.config = LaserData.DEFAULT_CONFIG if config is None else config
        if calibration is None:
            self.calibration = {
                k: LaserData.DEFAULT_CALIBRATION for k in self.data.dtype.names
            }
        else:
            self.calibration = calibration
        self.name = name
        self.source = source

    def isotopes(self) -> List[str]:
        return self.data.dtype.names

    def get(
        self,
        isotope: Optional[str] = None,
        calibrated: bool = False,
        trimmed: bool = False,
    ) -> np.ndarray:
        # Calibration
        if isotope is None:
            data = self.data
            if calibrated:
                for name in data.dtype.names:
                    gradient = self.calibration[name]["gradient"]
                    intercept = self.calibration[name]["intercept"]
                    data[name] = (data[name] - intercept) / gradient
        else:
            data = self.data[isotope]
            if calibrated:
                gradient = self.calibration[isotope]["gradient"]
                intercept = self.calibration[isotope]["intercept"]
                data = (data - intercept) / gradient
        # Trimming
        if trimmed:
            trim = self.config["trim"]
            if trim[1] > 0:
                data = data[:, trim[0] : -trim[1]]
            elif trim[0] > 0:
                data = data[:, trim[0] :]

        return data

    def pixelsize(self) -> Tuple[float, float]:
        return (self.config["speed"] * self.config["scantime"], self.config["spotsize"])

    def aspect(self) -> float:
        return self.config["spotsize"] / (
            self.config["speed"] * self.config["scantime"]
        )

    def extent(self, trimmed: bool = False) -> Tuple[int, int, int, int]:
        # Image data is stored [rows][cols]
        x_shape = self.data.shape[1]
        if trimmed:
            trim = self.config["trim"]
            x_shape -= trim[0] + trim[1]
        x = x_shape * self.pixelsize()[0]
        y = self.data.shape[0] * self.pixelsize()[1]
        return (0, x, 0, y)

    def setTrim(self, trim: Tuple[float, float], unit: str = "rows"):
        """Set the trim value using the provided unit.
        Valid units are 'rows', 'μm' and 's'."""
        if unit == "μm":
            width = self.pixelsize()[0]
            trim = (trim[0] / width, trim[1] / width)
        elif unit == "s":
            width = self.config["scantime"]
            trim = (trim[0] / width, trim[1] / width)
        self.config["trim"] = (int(trim[0]), int(trim[1]))

    def trimAs(self, unit: str) -> Tuple[int, int]:
        """Returns the trim in given unit.
        Valid units are 'rows', 'μm' and 's'."""
        trim = self.config["trim"]
        if unit == "μm":
            width = self.pixelsize()[0]
            trim = (trim[0] * width, trim[1] * width)
        elif unit == "s":
            width = self.config["scantime"]
            trim = (trim[0] * width, trim[1] * width)
        return trim

    def layers(self) -> int:
        return 1
