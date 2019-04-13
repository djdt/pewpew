import numpy as np
import copy

from pewpew.lib.laser.config import LaserConfig
from pewpew.lib.laser.data import LaserData

from typing import Dict, List, Tuple


class Laser(object):
    def __init__(
        self,
        data: Dict[str, LaserData] = None,
        config: LaserConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        self.data = data if data is not None else {}

        self.config = copy.copy(config) if config is not None else LaserConfig()

        self.name = name
        self.filepath = filepath

    def names(self) -> List[str]:
        return list(self.data.keys())

    def get(
        self,
        name: str,
        calibrate: bool = False,
        extent: Tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        # Calibration
        return self.data[name].get(self.config, calibrate=calibrate, extent=extent)

    def get_structured(
        self, calibrate: bool = False, extent: Tuple[float, float, float, float] = None
    ) -> np.ndarray:
        data = []
        for name in self.names():
            data.append(
                self.data[name].get(  # type: ignore
                    self.config, calibrate=calibrate, extent=extent
                )
            )
        dtype = [(name, float) for name in self.names()]
        structured = np.empty(data[0].shape, dtype)
        for name, d in zip(self.names(), data):
            structured[name] = d
        return structured

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

    # def extent(self) -> Tuple[float, float, float, float]:
    #     # Image data is stored [rows][cols]
    #     x = self.width() * self.config.pixel_width()
    #     y = self.height() * self.config.pixel_height()
    #     return (0.0, x, 0.0, y)

    @staticmethod
    def formatName(name: str) -> str:
        pass
