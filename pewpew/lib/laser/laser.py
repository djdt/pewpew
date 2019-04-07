import numpy as np
import copy

from typing import Dict, List, Tuple

from pewpew.lib.laser import LaserConfig, LaserData


class Laser(object):
    DEFAULT_CALIBRATION = {"gradient": 1.0, "intercept": 0.0, "unit": ""}
    DEFAULT_CONFIG = {"spotsize": 30.0, "speed": 120.0, "scantime": 0.25}

    def __init__(
        self,
        datas: Dict[str, LaserData] = None,
        config: LaserConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        self.datas = (
            datas
            if datas is not None
            else {"none", LaserData(np.zeros([1, 1], dtype=float), "none")}
        )
        # Copys of dicts are made
        self.config = config if config is not None else LaserConfig()
        self.name = name
        self.filepath = filepath

    def isotopes(self) -> List[str]:
        return self.datas.keys()

    def get(
        self,
        name: str = None,
        calibrated: bool = False,
        extent: Tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        # Calibration
        data = self.data[name].get(calibrated=calibrated)
        # Trimming
        if extent is not None:
            px, py = self.config.pixel_size()
            x1, x2 = int(extent[0] / px), int(extent[1] / px)
            y1, y2 = int(extent[2] / py), int(extent[3] / py)
            # We have to invert the extent, as mpl use bottom left y coords
            yshape = data.shape[0]
            data = data[yshape - y2 : yshape - y1, x1:x2]

        return data

    def convert(self, x: float, unit_from: str, unit_to: str) -> float:
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
        x = self.data.width() * self.config.pixel_width()
        y = self.data.height() * self.config.pixel_height()
        return (0.0, x, 0.0, y)

    def layers(self) -> int:
        return 1
