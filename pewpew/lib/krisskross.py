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
        trimmed: bool = False,
        flattened: bool = True,
    ) -> np.ndarray:
        data = super().get(isotope=isotope, calibrated=calibrated, trimmed=trimmed)
        if flattened:
            data = np.mean(data, axis=2)
        return data

    def split(self) -> List[LaserData]:
        lds = []
        for data in np.dsplit(self.data, self.data.shape[2]):
            # Strip the third dimension
            lds.append(
                KrissKrossData(
                    data=data,
                    config=self.config,
                    calibration=self.calibration,
                    source=self.source,
                )
            )
        return lds

    def extent(self, trimmed: bool = False) -> Tuple[float, float, float, float]:
        # Image data is stored [rows][cols]
        extent = super().extent(trimmed)
        return (0.0, extent[1], 0.0, extent[3] / self.aspect())

    def layers(self) -> int:
        return self.data.shape[2]
