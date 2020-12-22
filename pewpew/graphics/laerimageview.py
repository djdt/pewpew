"""This is a more efficient way (less paints get called.)"""
import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

import colortable
from npimage import NumpyImage

from overlaygraphics import OverlayScene, OverlayView
from overlayitems import ColorBarOverlay, MetricScaleBarOverlay, LabelOverlay
from viewoptions import ViewOptions


class LaserImageView(OverlayView):
    def __init__(self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None):
        self.viewoptions = viewoptions

        self._scene = OverlayScene(0, 0, 640, 480)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        self.label = LabelOverlay(
            "_", font=self.viewoptions.font, color=self.viewoptions.font_color
        )
        self.scalebar = MetricScaleBarOverlay(
            font=self.viewoptions.font, color=self.viewoptions.font_color
        )
        self.colorbar = ColorBarOverlay(
            [], 0, 1, font=self.viewoptions.font, color=self.viewoptions.font_color
        )

        self.image: QtGui.QImage = None
        self.pixmap: QtWidgets.QGraphicsPixmapItem = None

        super().__init__(self._scene, parent)

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
        # self.colorbar.setPos(0, -5)

    def drawData(self, data: np.ndarray, name: str) -> None:
        if self.pixmap is not None:
            self.scene().removeItem(self.pixmap)

        self.image = NumpyImage(data, 0.0, np.percentile(data, 95))
        self.image.setColorTable(colortable.to_table("turbo"))
        pixmap = QtGui.QPixmap.fromImage(self.image)
        self.pixmap = self.scene().addPixmap(pixmap)

        self.label.setVisible(self.viewoptions.items["label"])
        self.label.text = name

        self.scalebar.setVisible(self.viewoptions.items["scalebar"])

        self.colorbar.setVisible(self.viewoptions.items["colorbar"])
        self.colorbar.updateTable(
            self.image.colorTable(), self.image.vmin, self.image.vmax
        )

        self.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.viewChanged.emit(self.viewport().rect())
        self.viewport().update()
