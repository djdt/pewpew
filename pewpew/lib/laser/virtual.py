from .config import LaserConfig
from .data import LaserData

from typing import Callable, Tuple
import numpy as np


class VirtualData(LaserData):
    def __init__(self, data: LaserData, name: str, op: Callable, data2: LaserData):
        self.data = np.empty((1, 1), dtype=float)
        self.name = name

        self.data1 = data
        self.op = op
        self.data2 = data2

        self.gradient = self.op(self.data1.gradient, self.data2.gradient)
        self.intercept = self.op(self.data1.intercept, self.data2.intercept)
        self.unit = self.data1.unit

    def get(
        self,
        config: LaserConfig,
        calibrate: bool = False,
        extent: Tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        d1 = self.data1.get(config, calibrate=calibrate, extent=extent)
        d2 = self.data2.get(config, calibrate=calibrate, extent=extent)

        return self.op(d1, d2)
