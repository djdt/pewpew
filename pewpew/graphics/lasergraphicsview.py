"""This is a more efficient way (less paints get called.)"""
import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

from pewlib.laser import _Laser

from pewpew.graphics import colortable
from pewpew.graphics.numpyimage import NumpyImage
from pewpew.graphics.overlaygraphics import OverlayScene, OverlayView
from pewpew.graphics.overlayitems import (
    ColorBarOverlay,
    MetricScaleBarOverlay,
    LabelOverlay,
)
from pewpew.graphics.options import GraphicsOptions


class LaserGraphicsView(OverlayView):
    def __init__(self, options: GraphicsOptions, parent: QtWidgets.QWidget = None):
        self.options = options

        self._scene = OverlayScene(0, 0, 640, 480)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        super().__init__(self._scene, parent)

        self.image: NumpyImage = None

        self.label = LabelOverlay(
            "_", font=self.options.font, color=self.options.font_color
        )
        self.scalebar = MetricScaleBarOverlay(
            font=self.options.font, color=self.options.font_color
        )
        self.colorbar = ColorBarOverlay(
            [], 0, 1, font=self.options.font, color=self.options.font_color
        )

        self.scene().addOverlayItem(
            self.label,
            QtCore.Qt.TopLeftCorner,
            QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft,
        )
        self.label.setPos(50, 10)
        self.scene().addOverlayItem(
            self.scalebar,
            QtCore.Qt.TopRightCorner,
            QtCore.Qt.AlignTop | QtCore.Qt.AlignRight,
        )
        self.scalebar.setPos(0, 10)
        self.scene().addOverlayItem(
            self.colorbar,
            QtCore.Qt.BottomLeftCorner,
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft,
        )

    def drawImage(self, data: np.ndarray, rect: QtCore.QRectF, name: str) -> None:
        if self.image is not None:
            self.scene().removeItem(self.image)

        vmin, vmax = self.options.get_colorrange_as_float(name, data)
        self.image = NumpyImage(data, rect, vmin, vmax)
        self.image.image.setColorTable(colortable.to_table(self.options.colortable))
        self.colorbar.updateTable(
            self.image.image.colorTable(), self.image.vmin, self.image.vmax
        )

        self.scene().addItem(self.image)

        if self.sceneRect() != rect:
            self.setSceneRect(rect)
            self.fitInView(rect, QtCore.Qt.KeepAspectRatio)

        self.viewChanged.emit(self.viewport().rect())
        self.viewport().update()

    def drawLaser(self, laser: _Laser, name: str, layer: int = None) -> None:
        kwargs = {"calibrate": self.options.calibrate, "layer": layer, "flat": True}

        data = laser.get(name, **kwargs)
        unit = laser.calibration[name].unit if self.options.calibrate else ""

        # Get extent
        if laser.layers > 1:
            x0, x1, y0, y1 = laser.config.data_extent(data.shape, layer=layer)
        else:
            x0, x1, y0, y1 = laser.config.data_extent(data.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

        # Update overlay items
        self.label.text = name
        self.colorbar.unit = unit

        # Set overlay items visibility
        self.label.setVisible(self.options.items["label"])
        self.scalebar.setVisible(self.options.items["scalebar"])
        self.colorbar.setVisible(self.options.items["colorbar"])

        self.drawImage(data, rect, name)
