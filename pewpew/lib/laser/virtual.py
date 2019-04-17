from pewpew.lib.laser.config import LaserConfig
from pewpew.lib.laser.data import LaserData

from typing import Callable, Tuple
import numpy as np


class VirtualData(LaserData):
    SYMBOLS = {
        "add": "+",
        "divide": "/",
        "true_divide": "/",
        "multiply": "*",
        "subtract": "-",
        "greater": ">",
        "less": "<",
        "equal": "=",
        "not_equal": "!=",
    }

    def __init__(
        self,
        data1: LaserData,
        name: str = None,
        op: Callable = None,
        data2: LaserData = None,
        condition1: Tuple[Callable, float] = None,
        condition2: Tuple[Callable, float] = None,
        fill_value: float = -1.0,
    ):
        self.data = np.empty((1, 1), dtype=float)

        self.data1 = data1
        self.data2 = data2
        self.op = op
        self.condition1 = condition1
        self.condition2 = condition2
        self.fill_value = fill_value

        self.name = name if name is not None else self.generateName()

        self.gradient = 1.0
        self.intercept = 0.0
        self.unit = LaserData.DEFAULT_UNIT

    def generateName(self) -> str:
        name = self.data1.name
        if self.condition1 is not None:
            name += f"[{VirtualData.SYMBOLS[self.condition1[0].__name__]}{self.condition1[1]}]"
        if self.op is not None:
            name += f"{VirtualData.SYMBOLS[self.op.__name__]}"
        elif self.data2 is not None:
            name += "=>"
        if self.data2 is not None:
            name += self.data2.name
        if self.condition2 is not None:
            name += f"[{VirtualData.SYMBOLS[self.condition2[0].__name__]}{self.condition2[1]}]"
        return name

    def get(
        self,
        config: LaserConfig,
        calibrate: bool = False,
        extent: Tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        d1 = self.data1.get(config, calibrate=calibrate, extent=extent)
        if self.condition1 is not None:
            mask = self.condition1[0](d1, self.condition1[1])
            d1 = np.where(mask, d1, np.full_like(d1, self.fill_value))

        # If op and data2 are set then return d1 op d2
        if self.data2 is not None:
            d2 = self.data2.get(config, calibrate=calibrate, extent=extent)
            if self.condition2 is not None:
                mask = self.condition2[0](d2, self.condition2[1])
                d2 = np.where(mask, d2, np.full_like(d2, self.fill_value))
            if self.op is not None:
                return self.op(d1, d2)
            else:
                d1 = np.where(d2 != self.fill_value, d1, d2)

        return d1
