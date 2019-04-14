import numpy as np

from pewpew.lib.laser.config import LaserConfig

from typing import Tuple


class LaserData(object):
    DEFAULT_UNIT = ""

    def __init__(
        self,
        data: np.ndarray,
        name: str,
        gradient: float = 1.0,
        intercept: float = 0.0,
        unit: str = None,
    ):
        self.data = data
        self.name = name
        self.gradient = gradient
        self.intercept = intercept
        self.unit = unit if unit is not None else LaserData.DEFAULT_UNIT

    def get(
        self,
        config: LaserConfig,
        calibrate: bool = False,
        extent: Tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        data = self.data

        # Do this first to minimise required ops
        if extent is not None:
            px, py = config.pixel_size()
            x1, x2 = int(extent[0] / px), int(extent[1] / px)
            y1, y2 = int(extent[2] / py), int(extent[3] / py)
            # We have to invert the extent, as mpl use bottom left y coords
            yshape = data.shape[0]
            data = data[yshape - y2 : yshape - y1, x1:x2]

        if calibrate:
            data = (data - self.intercept) / self.gradient

        return data
