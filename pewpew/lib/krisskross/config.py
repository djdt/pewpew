import numpy as np
from fractions import Fraction
from pewpew.lib.laser import LaserConfig

from typing import List, Tuple


class KrissKrossConfig(LaserConfig):
    def __init__(
        self,
        spotsize: float = 10.0,
        speed: float = 10.0,
        scantime: float = 0.1,
        warmup: float = 10.0,
        pixel_offsets: List[Fraction] = [Fraction(0, 2), Fraction(1, 2)],
        horizontal_first: bool = True,
    ):
        super().__init__(spotsize=spotsize, speed=speed, scantime=scantime)
        self.warmup = warmup
        self.pixel_offsets = pixel_offsets
        self.horizontal_first = horizontal_first
        self._calculate_subpixel_params()

    def pixel_width(self) -> float:
        return (self.speed * self.scantime) / self.subpixel_per_pixel[0]

    def pixel_height(self) -> float:
        return (self.speed * self.scantime) / self.subpixel_per_pixel[1]

    def warmup_lines(self) -> int:
        return np.round(self.warmup / self.scantime).astype(int)

    def magnification_factor(self) -> float:
        return np.round(self.spotsize / (self.speed * self.scantime)).astype(int)

    def subpixel_offsets(self) -> List[int]:
        return [offset // self.subpixel_gcd for offset in self.pixel_offsets]

    def _calculate_subpixel_params(self) -> None:
        gcd = np.gcd.reduce(self.pixel_offsets)
        denom = (
            Fraction(gcd * self.magnification_factor()).limit_denominator().denominator
        )
        self.subpixel_gcd: Fraction = gcd
        self.subpixel_per_pixel: Tuple[int, int] = (denom, denom)
