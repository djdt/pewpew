import numpy as np

from typing import Dict, List, Tuple

from pewpew.lib.laser import LaserConfig, LaserData


class Laser(object):
    def __init__(
        self,
        data: Dict[str, np.ndarray] = None,
        config: LaserConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        self.data: Dict[str, LaserData] = {}
        self.width = 0
        self.height = 0
        self.depth = 0
        if data is not None:
            for k, v in data.items():
                self.add_data(k, v)

        self.config = config if config is not None else LaserConfig()

        self.name = name
        self.filepath = filepath

    def add_data(self, name: str, x: np.ndarray) -> None:
        data = LaserData(x, name)
        if self.width == 0 or self.height == 0 or self.depth == 0:
            self.width = data.width()
            self.height = data.height()
            self.depth = data.depth()
        else:
            assert self.width == data.width()
            assert self.height == data.height()
            assert self.depth == data.depth()
        self.data[name] = data

    def names(self) -> List[str]:
        return list(self.data.keys())

    def get(
        self,
        name: str,
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
            x = x / self.config.scantime
        elif unit_from in ["um", "μm", "micro meters"]:
            x = x / self.config.pixel_width()
        # Convert to desired unit
        if unit_to in ["s", "seconds"]:
            x = x * self.config.scantime
        elif unit_to in ["um", "μm", "micro meters"]:
            x = x * self.config.pixel_width()
        return x

    def extent(self) -> Tuple[float, float, float, float]:
        # Image data is stored [rows][cols]
        x = self.width * self.config.pixel_width()
        y = self.height * self.config.pixel_height()
        return (0.0, x, 0.0, y)
