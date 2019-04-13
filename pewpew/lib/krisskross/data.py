import numpy as np
from pewpew.lib.calc import subpixel_offset_equal
from pewpew.lib.laser.data import LaserData
from pewpew.lib.krisskross.config import KrissKrossConfig
from pewpew.lib.laser.config import LaserConfig

from typing import List, Tuple


class KrissKrossData(LaserData):
    def __init__(
        self,
        data: List[np.ndarray],
        name: str,
        gradient: float = 1.0,
        intercept: float = 0.0,
        unit: str = None,
    ):
        super().__init__(data, name, gradient=gradient, intercept=intercept, unit=unit)

    def _krisskross(self, config: KrissKrossConfig) -> np.ndarray:
        warmup = config.warmup_lines()
        mfactor = config.magnification_factor()

        j = 0 if config.horizontal_first else 1
        # Calculate the line lengths
        length = (self.data[1].shape[0] * mfactor, self.data[0].shape[0] * mfactor)
        # Reshape the layers and stack into matrix
        aligned = np.empty(
            (length[1], length[0], len(self.data)), dtype=self.data[0].dtype
        )
        for i, layer in enumerate(self.data):
            # Trim data of warmup time and excess
            layer = layer[:, warmup : warmup + length[(i + j) % 2]]
            # Stretch array
            layer = np.repeat(layer, mfactor, axis=0)
            # Flip vertical layers
            if (i + j) % 2 == 1:
                layer = layer.T
            aligned[:, :, i] = layer

        return subpixel_offset_equal(
            aligned, config.subpixel_offsets(), config.subpixel_per_pixel[0]
        )

    def get(
        self,
        config: LaserConfig,
        calibrate: bool = False,
        extent: Tuple[float, float, float, float] = None,
        flat: bool = False,
        layer: int = None,
    ) -> np.ndarray:

        assert isinstance(config, KrissKrossConfig)

        if layer is None:
            data = self._krisskross(config)
        else:
            data = self.data[layer]

        # Do this first to minimise required ops
        if extent is not None:
            px, py = config.pixel_size()
            x1, x2 = int(extent[0] / px), int(extent[1] / px)
            y1, y2 = int(extent[2] / py), int(extent[3] / py)
            # We have to invert the extent, as mpl use bottom left y coords
            yshape = data.shape[0]
            data = data[yshape - y2 : yshape - y1, x1:x2]

        if layer is None and flat:
            data = np.mean(data, axis=2)

        if calibrate:
            data = (data - self.intercept) / self.gradient

        return data
