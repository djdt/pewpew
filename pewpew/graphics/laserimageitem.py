from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from pewlib.laser import Laser
from pewlib.srr.config import SRRConfig

from pewpew.lib.numpyqt import array_to_image

from pewpew.graphics import colortable
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.imageitems import ScaledImageItem
# from pewpew.graphics.overlayitems import OverlayItem, LabelOverlay

from typing import Optional


class LaserImageItem(ScaledImageItem):
    def __init__(
        self,
        laser: Laser,
        options: GraphicsOptions,
        current_element: Optional[str] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        self.laser = laser
        self.options = options
        self.current_element = current_element or self.laser.elements[0]

        super().__init__(
            self.laserImage(self.current_element),
            self.laserRect(),
            smooth=self.options.smoothing,
            snap=True,
            parent=parent,
        )

        # Name in top left corner
        # self.label_overlay = OverlayItem(LabelOverlay(self.laser.info["Name"], font=self.options.font, parent=self))
        # self.label.setTransformOriginPoint(self.transformOriginPoint().transposed())
        # self.label.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        print(self.label.boundingRect())
        if self.label.boundingRect().contains(event.scenePos()):
            self.label.mouseDoubleClickEvent(event)
        super().mouseDoubleClickEvent(event)

    def laserImage(self, name: str) -> QtGui.QImage:
        data = self.laser.get(name, calibrate=self.options.calibrate, flat=True)
        self.raw_data = np.ascontiguousarray(data)
        # unit = self.laser.calibration[name].unit if options.calibrate else ""

        vmin, vmax = self.options.get_color_range_as_float(name, self.raw_data)
        table = colortable.get_table(self.options.colortable)

        data = np.clip(self.raw_data, vmin, vmax)
        if vmin != vmax:  # Avoid div 0
            data = (data - vmin) / (vmax - vmin)

        image = array_to_image(data)
        image.setColorTable(table)
        return image

    def laserRect(self) -> QtCore.QRectF:
        x0, x1, y0, y1 = self.laser.config.data_extent(self.laser.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        rect.moveTopLeft(QtCore.QPointF(0, 0))
        return rect

    # def updateRect(self) -> None:
    #     x0, x1, y0, y1 = self.laser.config.data_extent(data.shape)

    #     rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
    #     # Recoordinate the top left to previous for correct updating
    #     rect.moveTopLeft(self.rect.topLeft())
    #     self.rect = QtCore.QRectF(rect)  # copy the rect
    # def getImage(self) -> QtGui.QImage:


    def refresh(self) -> None:
        self.rect = self.laserRect()
        image = self.laserImage(self.current_element)

        if self.options.smoothing:
            self.image = image.scaledToHeight(
                image.height() * 2, QtCore.Qt.SmoothTransformation
            )
            self.image_scale = 2
        else:
            self.image = image
            self.image_scale = 1
