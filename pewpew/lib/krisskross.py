import numpy as np

from pewpew.lib.laser import LaserData

from typing import List, Tuple


def krissKrossLayers(
    layers: List[np.ndarray], aspect: float, warmup: int, horizontal_first: bool = True
) -> np.ndarray:

    j = 0 if horizontal_first else 1
    aspect = int(aspect)
    trim = int(aspect / 2)
    # Calculate the line lengths
    length = (layers[1].shape[0] * aspect, layers[0].shape[0] * aspect)

    # Reshape the layers and stack into matrix
    transformed = []
    for i, layer in enumerate(layers):
        # Trim data of warmup time and excess
        layer = layer[:, warmup : warmup + length[(i + j) % 2]]
        # Stretch array
        layer = np.repeat(layer, aspect, axis=0)
        # Flip vertical layers and trim
        if (i + j) % 2 == 1:
            layer = layer.T
            layer = layer[trim:, trim:]
        elif trim > 0:
            layer = layer[:-trim, :-trim]

        transformed.append(layer)

    data = np.dstack(transformed)

    return data


class KrissKrossData(LaserData):
    def __init__(
        self,
        data: np.ndarray = None,
        config: dict = None,
        calibration: dict = None,
        name: str = "",
        source: str = "",
    ):
        super().__init__(
            data=data, config=config, calibration=calibration, name=name, source=source
        )

    def fromLayers(
        self,
        layers: List[np.ndarray],
        warmup_time: float = 13.0,
        horizontal_first: bool = True,
    ) -> None:
        warmup = int(warmup_time / self.config["scantime"])
        self.data = krissKrossLayers(layers, self.aspect(), warmup, horizontal_first)

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

    def extent(self) -> Tuple[float, float, float, float]:
        # Image data is stored [rows][cols]
        extent = super().extent()
        return (0.0, extent[1], 0.0, extent[3] / self.aspect())

    def layers(self) -> int:
        return self.data.shape[2]

def krissKrossLayers2(
    layers: List[np.ndarray], aspect: float, warmup: int, horizontal_first: bool = True
) -> np.ndarray:

    j = 0 if horizontal_first else 1
    subpixel = 0.5, 0.5
    sub_per_pixel = [int(1000.0 / np.gcm(1000, int(sp * 1000))) for sp in subpixel]
    offset = subpixel[0] * sub_per_pixel[0], subpixel[1] * sub_per_pixel[1]

    aspect = int(aspect)
    trim = int(aspect / 2)
    # Calculate the line lengths
    length = (layers[1].shape[0] * aspect, layers[0].shape[0] * aspect)

    # Reshape the layers and stack into matrix
    transformed = []
    for i, layer in enumerate(layers):
        # Trim data of warmup time and excess
        layer = layer[:, warmup : warmup + length[(i + j) % 2]]
        # Stretch array
        layer = np.repeat(layer, aspect, axis=0)  # Here we would repeat by the overlap
        # Flip vertical layers and trim
        if (i + j) % 2 == 1:
            layer = layer.T
            layer = layer[trim:, trim:]
        elif trim > 0:
            layer = layer[:-trim, :-trim]

        transformed.append(layer)

    data = np.dstack(transformed)

    return data


if __name__ == "__main__":
    pass
