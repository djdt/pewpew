import numpy as np
import copy

from typing import List, Tuple


class LaserData(object):
    DEFAULT_CALIBRATION = {"gradient": 1.0, "intercept": 0.0, "unit": ""}
    DEFAULT_CONFIG = {"spotsize": 30.0, "speed": 120.0, "scantime": 0.25}

    def __init__(
        self,
        data: np.ndarray = None,
        config: dict = None,
        calibration: dict = None,
        name: str = "",
        source: str = "",
    ):
        self.data = (
            np.array([[0]], dtype=[("none", np.float64)]) if data is None else data
        )
        # Copys of dicts are made
        self.config: dict = copy.deepcopy(
            LaserData.DEFAULT_CONFIG if config is None else config
        )
        if calibration is None:
            self.calibration = {
                k: dict(LaserData.DEFAULT_CALIBRATION) for k in self.data.dtype.names
            }
        else:
            self.calibration = copy.deepcopy(calibration)
        self.name = name
        self.source = source

    def isotopes(self) -> List[str]:
        return self.data.dtype.names

    def get(
        self,
        isotope: str = None,
        calibrated: bool = False,
        extent: Tuple[float, float, float, float] = None,
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
        if extent is not None:
            pixel = self.pixelsize()
            x1, x2 = int(extent[0] / pixel[0]), int(extent[1] / pixel[0])
            y1, y2 = int(extent[2] / pixel[1]), int(extent[3] / pixel[1])
            # We have to invert the extent, as mpl use bottom left y coords
            yshape = data.shape[0]
            data = data[yshape - y2 : yshape - y1, x1:x2]

        return data

    def convertRow(self, x: float, unit_from: str, unit_to: str) -> float:
        # Convert into rows
        if unit_from in ["s", "seconds"]:
            x = x / self.config["scantime"]
        elif unit_from in ["um", "μm", "micro meters"]:
            x = x / (self.config["speed"] * self.config["scantime"])
        # Convert to desired unit
        if unit_to in ["s", "seconds"]:
            x = x * self.config["scantime"]
        elif unit_to in ["um", "μm", "micro meters"]:
            x = x * (self.config["speed"] * self.config["scantime"])
        return x

    def pixelsize(self) -> Tuple[float, float]:
        return (self.config["speed"] * self.config["scantime"], self.config["spotsize"])

    def aspect(self) -> float:
        return self.config["spotsize"] / (
            self.config["speed"] * self.config["scantime"]
        )

    def extent(self) -> Tuple[float, float, float, float]:
        # Image data is stored [rows][cols]
        x = self.data.shape[1] * self.pixelsize()[0]
        y = self.data.shape[0] * self.pixelsize()[1]
        return (0.0, x, 0.0, y)

    def layers(self) -> int:
        return 1


class KrissKrossData(LaserData):
    def pixelsize(self) -> Tuple[float, float]:
        return (
            self.config["speed"] * self.config["scantime"],
            self.config["spotsize"] / self.aspect(),
        )
