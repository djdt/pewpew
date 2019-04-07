import numpy as np


class LaserData(object):
    DEFAULT_UNIT = "<None>"

    def __init__(
        self,
        data: np.ndarray,
        name: str,
        gradient: float = 1.0,
        intercept: float = 0.0,
        unit: str = None,
    ):
        self.data = data
        self.name = name
        self.gradient = gradient
        self.intercept = intercept
        self.unit = unit if unit is not None else LaserData.DEFAULT_UNIT

    def width(self) -> int:
        return self.data.shape[1]

    def height(self) -> int:
        return self.data.shape[0]

    def depth(self) -> int:
        return self.data.shape[2] if self.data.ndim > 2 else 1

    def get(self, calibrated: bool = True) -> np.ndarray:
        if calibrated:
            return self.calibrate()
        else:
            return self.data

    def calibrate(self) -> np.ndarray:
        return (self.data - self.intercept) / self.gradient
