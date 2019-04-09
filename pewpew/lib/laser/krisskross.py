import numpy as np
from fractions import Fraction

from pewpew.lib.laser import Laser, KrissKrossConfig
from pewpew.lib.calc import subpixel_offset_equal

from typing import List, Tuple, Type, TypeVar


def krissKrossLayers(
    layers: List[np.ndarray],
    aspect: int,
    warmup: int,
    offsets: List[Fraction],
    horizontal_first: bool = True,
) -> Tuple[np.ndarray, int]:

    j = 0 if horizontal_first else 1
    # Calculate the line lengths
    length = (layers[1].shape[0] * aspect, layers[0].shape[0] * aspect)

    # Reshape the layers and stack into matrix
    aligned = np.empty((length[1], length[0], len(layers)), dtype=layers[0].dtype)
    for i, layer in enumerate(layers):
        # Trim data of warmup time and excess
        layer = layer[:, warmup : warmup + length[(i + j) % 2]]
        # Stretch array
        layer = np.repeat(layer, aspect, axis=0)
        # Flip vertical layers
        if (i + j) % 2 == 1:
            layer = layer.T
        aligned[:, :, i] = layer

    # Calculate the require pixel stretch
    gcd = np.gcd.reduce(offsets)
    stretch = (gcd * aspect).limit_denominator().denominator

    return subpixel_offset_equal(aligned, [o // gcd for o in offsets], stretch), stretch


KKType = TypeVar("KKType", bound="KrissKrossData")  # For typing


class KrissKrossData(Laser):
    def __init__(
        self,
        data: np.ndarray = None,
        config: KrissKrossConfig = None,
        calibration: dict = None,
        name: str = "",
        source: str = "",
    ):
        self.stretch = (1, 1)
        super().__init__(
            data=data, config=config, calibration=calibration, name=name, source=source
        )

    def get(
        self,
        isotope: str = None,
        calibrated: bool = False,
        extent: Tuple[float, float, float, float] = None,
        flattened: bool = True,
    ) -> np.ndarray:
        data = super().get(isotope=isotope, calibrated=calibrated, extent=extent)
        if flattened:
            data = np.mean(data, axis=2)
        return data

    def pixelsize(self) -> Tuple[float, float]:
        return (
            (self.config["speed"] * self.config["scantime"]) / self.stretch[0],
            self.config["spotsize"] / (self.aspect() * self.stretch[1]),
        )

    def layers(self) -> int:
        return self.data.shape[2]

    @classmethod
    def fromLayers(
        cls: Type[KKType],
        layers: List[np.ndarray],
        config: dict,
        calibration: dict = None,
        name: str = "",
        source: str = "",
        offsets: List[Fraction] = [Fraction(0, 2), Fraction(1, 2)],
        warmup_time: float = 12.0,
        horizontal_first: bool = True,
    ) -> KKType:
        warmup = int(warmup_time / config["scantime"])
        aspect = int(config["spotsize"] / (config["speed"] * config["scantime"]))
        data, stretch = krissKrossLayers(
            layers, aspect, warmup, offsets, horizontal_first
        )
        kkd = cls(data, config, calibration, name, source)
        kkd.stretch = (stretch, stretch)
        return kkd
