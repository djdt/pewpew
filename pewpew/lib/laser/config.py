from typing import Dict, Tuple


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
