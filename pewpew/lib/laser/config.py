import numpy as np

from typing import Tuple, List
from fractions import Fraction


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
        warmup: float = 10.0,
        subpixel_per_pixel: Tuple[int, int] = (1, 1),
    ):
        super().__init__(spotsize=spotsize, speed=speed, scantime=scantime)
        self.warmup = warmup
        self.subpixel_per_pixel = subpixel_per_pixel

    def pixel_width(self) -> float:
        return (self.speed * self.scantime) / self.subpixel_per_pixel[0]

    def pixel_height(self) -> float:
        return (self.speed * self.scantime) / self.subpixel_per_pixel[1]

    def warmup_lines(self) -> int:
        return np.round(self.warmup / self.scantime).astype(int)

    def layer_aspect(self) -> float:
        return self.spotsize / (self.speed * self.scantime)

    def calculate_subpixel_per_pixel(self, offsets: List[Fraction]) -> Tuple[int, int]:
        gcd = np.gcd.reduce(offsets)
        denom = (
            Fraction(gcd * np.round(self.layer_aspect()))
            .limit_denominator()
            .denominator
        )
        self.subpixel_per_pixel = (denom, denom)
        return self.subpixel_per_pixel
