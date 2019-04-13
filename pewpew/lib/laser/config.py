from typing import Tuple
import numpy as np


class LaserConfig(object):
    def __init__(
        self, spotsize: float = 30.0, speed: float = 120.0, scantime: float = 0.25
    ):
        self.spotsize = spotsize
        self.speed = speed
        self.scantime = scantime

    def pixel_width(self) -> float:
        return self.speed * self.scantime

    def pixel_height(self) -> float:
        return self.spotsize

    def aspect(self) -> float:
        return self.pixel_height() / self.pixel_width()

    def pixel_size(self) -> Tuple[float, float]:
        return (self.pixel_width(), self.pixel_height())

    def data_extent(self, data: np.ndarray) -> Tuple[float, float, float, float]:
        return (
            0.0,
            self.pixel_width() * data.shape[1],
            0.0,
            self.pixel_height() * data.shape[0],
        )
