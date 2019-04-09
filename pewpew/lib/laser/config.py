from typing import Tuple


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


class KrissKrossConfig(LaserConfig):
    def __init__(
        self,
        spotsize: float = 10.0,
        speed: float = 10.0,
        scantime: float = 0.1,
        pixel_stretch: Tuple[int, int] = (1, 1)
    ):
        super().__init__(spotsize=spotsize, speed=speed, scantime=scantime)
        self.pixel_stretch = pixel_stretch

    def pixel_width(self) -> float:
        return (self.speed * self.scantime) / self.pixel_stretch[0]

    def pixel_height(self) -> float:
        return (self.speed * self.scantime) / self.pixel_stretch[1]
