from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from pewlib.laser import Laser
from pewlib.srr.config import SRRConfig

from pewpew.lib.numpyqt import array_to_image

from pewpew.graphics import colortable
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.imageitems import ScaledImageItem

from typing import Optional


class LaserImageItem(ScaledImageItem):
    def __init__(
        self,
        laser: Laser,
        # options: GraphicsOptions,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(QtGui.QImage(), QtCore.QRectF(0, 0, 1, 1), parent)

        self.laser = laser
        # self.options = options

    def updateImage(
        self, name: str, options: GraphicsOptions, layer: Optional[int] = None
    ) -> None:
        kwargs = {"calibrate": options.calibrate, "layer": layer, "flat": True}

        data = self.laser.get(name, **kwargs)
        self.raw_data = np.ascontiguousarray(data)
        # unit = self.laser.calibration[name].unit if options.calibrate else ""

        # Get extent
        if isinstance(self.laser.config, SRRConfig) and layer is not None:
            x0, x1, y0, y1 = self.laser.config.data_extent(data.shape, layer=layer)
        else:
            x0, x1, y0, y1 = self.laser.config.data_extent(data.shape)

        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        # Recoordinate the top left to previous for correct updating
        rect.moveTopLeft(self.rect.topLeft())
        self.rect = QtCore.QRectF(rect)  # copy the rect

        vmin, vmax = options.get_color_range_as_float(name, self.raw_data)
        table = colortable.get_table(options.colortable)

        data = np.clip(self.raw_data, vmin, vmax)
        if vmin != vmax:  # Avoid div 0
            data = (data - vmin) / (vmax - vmin)

        image = array_to_image(data)
        image.setColorTable(table)

        if options.smoothing:
            self.image = image.scaledToHeight(
                image.height() * 2, QtCore.Qt.SmoothTransformation
            )
            self.image_scale = 2
        else:
            self.image = image
            self.image_scale = 1
