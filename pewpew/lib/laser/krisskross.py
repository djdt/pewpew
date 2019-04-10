import numpy as np
from fractions import Fraction

from pewpew.lib.laser import Laser
from pewpew.lib.laser.config import KrissKrossConfig
from pewpew.lib.laser.data import KrissKrossData
from pewpew.lib.calc import subpixel_offset_equal

from typing import Dict, List, Tuple, Type, TypeVar


def krisskross_layers(
    layers: List[np.ndarray],
    warmup: int,
    aspect: int,
    stretch: int,
    offsets: List[Fraction],
    horizontal_first: bool = True,
) -> np.ndarray:

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

    # Calculate the offset gcd
    gcd = np.gcd.reduce(offsets)
    return subpixel_offset_equal(aligned, [o // gcd for o in offsets], stretch)


KKType = TypeVar("KKType", bound="KrissKross")  # For typing


class KrissKross(Laser):
    def __init__(
        self,
        data: Dict[str, KrissKrossData] = None,
        config: KrissKrossConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        if config is None:
            config = KrissKrossConfig()

        super().__init__(
            data=data, config=config, name=name, filepath=filepath
        )  # type: ignore

    def get(
        self,
        name: str,
        calibrated: bool = False,
        extent: Tuple[float, float, float, float] = None,
        flat: bool = True,
    ) -> np.ndarray:
        # Calibration
        data = super().get(name, calibrated=calibrated)
        if flat:
            data = np.mean(data, axis=2)

        return data

    @classmethod
    def from_layers(
        cls: Type[KKType],
        layers: List[Laser],
        config: KrissKrossConfig,
        name: str = "",
        filepath: str = "",
        offsets: List[Fraction] = [Fraction(0, 2), Fraction(1, 2)],
        horizontal_first: bool = True,
    ) -> KKType:
        data = {}
        warmup = config.warmup_lines()
        aspect = np.round(config.layer_aspect()).astype(int)
        stretch = config.calculate_subpixel_per_pixel(offsets)[0]
        for name in layers[0].names():
            data[name] = KrissKrossData(
                krisskross_layers(
                    [l.data[name].data for l in layers],
                    warmup,
                    aspect,
                    stretch,
                    offsets,
                    horizontal_first,
                ),
                name,
                gradient=layers[0].data[name].gradient,
                intercept=layers[0].data[name].intercept,
                unit=layers[0].data[name].unit,
            )
        kkd = cls(data, config, name, filepath)
        return kkd
