import numpy as np
from fractions import Fraction

from pewpew.lib.laser import LaserData

from typing import List, Tuple


def subpixelOffset(
    images: List[np.ndarray], offsets: List[Tuple[int, int]], pixelsize: Tuple[int, int]
) -> np.ndarray:
    if offsets[0] != (0, 0):  # The zero offset
        offsets.insert(0, (0, 0))
    overlap = np.max(offsets, axis=0)
    shape = images[0].shape
    dtype = images[0].dtype

    for img in images:
        if img.shape != shape:
            raise ValueError("Arrays must have same shape.")
        if img.dtype != dtype:
            raise ValueError("Arrays must have same dtype.")

    new_shape = np.array(shape) * pixelsize + overlap
    data = np.zeros((*new_shape, len(images)), dtype=dtype)
    for i, img in enumerate(images):
        start = offsets[i % len(offsets)]
        end = -(overlap[0] - start[0]) or None, -(overlap[1] - start[1]) or None
        data[start[0] : end[0], start[1] : end[1], i] = np.repeat(
            img, pixelsize[0], axis=0
        ).repeat(pixelsize[1], axis=1)

    return data


def subpixelEqualOffset(
    images: List[np.ndarray], offsets: List[int], pixelsize: int
) -> np.ndarray:
    return subpixelOffset(images, [(o, o) for o in offsets], (pixelsize, pixelsize))


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
    aligned = []
    for i, layer in enumerate(layers):
        # Trim data of warmup time and excess
        layer = layer[:, warmup : warmup + length[(i + j) % 2]]
        # Stretch array
        layer = np.repeat(layer, aspect, axis=0)
        # Flip vertical layers
        if (i + j) % 2 == 1:
            layer = layer.T
        aligned.append(layer)

    # Calculate the require pixel stretch
    gcd = np.gcd.reduce(offsets)
    stretch = (gcd * aspect).limit_denominator().denominator

    return subpixelEqualOffset(aligned, [o // gcd for o in offsets], stretch), stretch


class KrissKrossData(LaserData):
    def __init__(
        self,
        data: np.ndarray = None,
        config: dict = None,
        calibration: dict = None,
        name: str = "",
        source: str = "",
    ):
        self.stretch = (1, 1)
        super().__init__(
            data=data, config=config, calibration=calibration, name=name, source=source
        )

    def fromLayers(
        self,
        layers: List[np.ndarray],
        offsets: List[Fraction],
        warmup_time: float = 13.0,
        horizontal_first: bool = True,
    ) -> None:
        warmup = int(warmup_time / self.config["scantime"])
        self.data, s = krissKrossLayers(
            layers, int(self.aspect()), warmup, offsets, horizontal_first
        )
        self.stretch = (s, s)

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
            self.config["speed"] * self.config["scantime"] / self.stretch[0],
            self.config["spotsize"] / self.aspect() / self.stretch[1],
        )

    def layers(self) -> int:
        return self.data.shape[2]
