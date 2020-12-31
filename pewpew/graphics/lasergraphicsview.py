"""This is a more efficient way (less paints get called.)"""
import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

from pewlib.laser import _Laser

from pewpew.graphics import colortable
from pewpew.graphics.items import (
    ScaledImageItem,
    ScaledImageSelectionItem,
    LassoImageSelectionItem,
    RectImageSelectionItem,
)
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.overlaygraphics import OverlayScene, OverlayView
from pewpew.graphics.overlayitems import (
    ColorBarOverlay,
    MetricScaleBarOverlay,
    LabelOverlay,
)
from pewpew.lib.numpyqt import array_as_indexed8


class LaserGraphicsView(OverlayView):
    def __init__(self, options: GraphicsOptions, parent: QtWidgets.QWidget = None):
        self.options = options
        self.data: np.ndarray = None
        self.mask: np.ndarray = None

        self._scene = OverlayScene(0, 0, 640, 480)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        super().__init__(self._scene, parent)
        self.cursors["selection"] = QtCore.Qt.ArrowCursor

        self.image: ScaledImageItem = None
        self.selection_item: ScaledImageSelectionItem = None
        self.selection_image: ScaledImageItem = None

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
        self.label.setPos(10, 10)
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

    def startLassoSelection(self) -> None:
        if self.selection_item is not None:
            self.scene().removeItem(self.selection_item)

        self.selection_item = LassoImageSelectionItem(self.image)
        self.selection_item.selectionChanged.connect(self.drawSelectionImage)
        self.scene().addItem(self.selection_item)
        self.selection_item.grabMouse()
        self.setInteractionMode("selection")

    def startRectangleSelection(self) -> None:
        if self.selection_item is not None:
            self.scene().removeItem(self.selection_item)

        self.selection_item = RectImageSelectionItem(self.image)
        self.selection_item.selectionChanged.connect(self.drawSelectionImage)
        self.scene().addItem(self.selection_item)
        self.selection_item.grabMouse()
        self.setInteractionMode("selection")

    def drawImage(self, data: np.ndarray, rect: QtCore.QRectF, name: str) -> None:
        if self.image is not None:
            self.scene().removeItem(self.image)

        self.data = data
        vmin, vmax = self.options.get_colorrange_as_float(name, self.data)
        table = colortable.get_table(self.options.colortable)

        array = array_as_indexed8(self.data, vmin, vmax)

        self.image = ScaledImageItem.fromArray(
            array, rect, QtGui.QImage.Format_Indexed8, colortable=table
        )
        self.scene().addItem(self.image)

        self.colorbar.updateTable(table, vmin, vmax)

        if self.sceneRect() != rect:
            self.setSceneRect(rect)
            self.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def drawSelectionImage(self) -> None:
        if self.selection_image is not None:
            self.scene().removeItem(self.selection_image)

        if self.selection_item is None:
            self.mask = np.ones(self.data.shape, dtype=np.bool)
        else:
            color = QtGui.QColor(255, 255, 255, a=128)
            self.selection_image = self.selection_item.maskAsImage(color)
            self.mask = self.selection_item.mask
            self.scene().addItem(self.selection_image)

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
