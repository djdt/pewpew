import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.graphics.imageitems import (
    SnapImageItem,
    RulerWidgetItem,
    ImageSliceWidgetItem,
)
from pewpew.graphics.selectionitems import (
    LassoImageSelectionItem,
    RectImageSelectionItem,
    SnapImageSelectionItem,
)
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.overlaygraphics import OverlayScene, OverlayView
from pewpew.graphics.overlayitems import (
    ColorBarOverlay,
    MetricScaleBarOverlay,
    LabelOverlay,
)

from typing import List, Optional


class LaserGraphicsView(OverlayView):
    """The pewpew laser view.

    Displays the image with correct scaling and an overlay label, sizebar and colorbar.
    If a selection is made the 'mask' is updated and a highlight is applied to sselected pixels.
    """

    cursorValueChanged = QtCore.Signal(float, float, float)

    def __init__(self, options: GraphicsOptions, parent: Optional[QtWidgets.QWidget] = None):
        self.options = options
        self.data: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None

        self._scene = OverlayScene(0, 0, 640, 480)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        super().__init__(self._scene, parent)
        self.cursors["selection"] = QtCore.Qt.ArrowCursor

        self.widget: Optional[QtWidgets.QGraphicsItem] = None

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

    def mapToData(self, pos: QtCore.QPointF) -> QtCore.QPoint:
        """Maps point to image pixel."""
        if self.image is None:
            return QtCore.QPoint(0, 0)

        return self.image.mapToData(pos)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        pos = self.mapToScene(event.pos())
        # if (
        #     self.image is not None
        #     and self.image.rect.left() < pos.x() < self.image.rect.right()
        #     and self.image.rect.top() < pos.y() < self.image.rect.bottom()
        # ):
        #     dpos = self.mapToData(pos)
        #     self.cursorValueChanged.emit(
        #         pos.x(), pos.y(), self.data[dpos.y(), dpos.x()]
        #     )
        # else:
        #     self.cursorValueChanged.emit(pos.x(), pos.y(), np.nan)

    def startLassoSelection(self) -> None:
        """Select image pixels using a lasso."""
        for item in self.items():
            if isinstance(item, SnapImageSelectionItem):
                self.scene().removeItem(item)

        selection_item = LassoImageSelectionItem(parent=None)
        # selection_item.selectionChanged.connect(self.drawSelectionImage)
        self.scene().addItem(selection_item)
        selection_item.grabMouse()
        self.setInteractionFlag("selection")

    def startRectangleSelection(self) -> None:
        """Select image pixels using a rectangle."""
        for item in self.items():
            if isinstance(item, SnapImageSelectionItem):
                self.scene().removeItem(item)

        selection_item = RectImageSelectionItem(parent=None)
        # self.selection_item.selectionChanged.connect(self.drawSelectionImage)
        self.scene().addItem(selection_item)
        selection_item.grabMouse()
        self.setInteractionFlag("selection")

    def endSelection(self) -> None:
        """End selection and remove highlight."""
        if self.selection_item is not None:
            self.selection_item = None
            self.scene().removeItem(self.selection_item)
        if self.selection_image is not None:
            self.scene().removeItem(self.selection_image)
            self.selection_image = None

        self.mask = np.zeros(self.data.shape, dtype=bool)
        self.setInteractionFlag("selection", False)

    def posInSelection(self, pos: QtCore.QPointF) -> bool:
        """Is the pos in the selected area."""
        if self.mask is None:
            return False
        pos = self.mapToData(self.mapToScene(pos))
        return self.mask[pos.y(), pos.x()]

    def startRulerWidget(self) -> None:
        """Measure distances using a ruler."""
        if self.image is None:
            return
        if self.widget is not None:
            self.scene().removeItem(self.widget)
        self.widget = RulerWidgetItem(self.image, font=self.options.font)
        self.widget.setZValue(1)
        self.scene().addItem(self.widget)
        self.widget.grabMouse()
        self.setInteractionFlag("widget")

    def startSliceWidget(self) -> None:
        """Display 1d slices in image."""
        if self.image is None or self.data is None:
            return
        if self.widget is not None:
            self.scene().removeItem(self.widget)
        self.widget = ImageSliceWidgetItem(
            self.image, self.data, font=self.options.font
        )
        self.widget.setZValue(1)
        self.scene().addItem(self.widget)
        self.widget.grabMouse()
        self.setInteractionFlag("widget")

    def endWidget(self) -> None:
        """End and remove any widgets."""
        if self.widget is not None:
            self.scene().removeItem(self.widget)
        self.widget = None
        self.setInteractionFlag("widget", False)

    def setOverlayItemVisibility(
        self, label: Optional[bool] = None, scalebar: Optional[bool] = None, colorbar: Optional[bool] = None
    ):
        """Set visibility of overlay items."""
        if label is None:
            label = self.options.overlay_items["label"]
        if scalebar is None:
            scalebar = self.options.overlay_items["scalebar"]
        if colorbar is None:
            colorbar = self.options.overlay_items["colorbar"]

        self.label.setVisible(label)
        self.scalebar.setVisible(scalebar)
        self.colorbar.setVisible(colorbar)

    def zoomReset(self) -> None:
        rect = QtCore.QRectF(0, 0, 0, 0)
        for item in self.scene().items():
            if isinstance(item, SnapImageItem):
                rect = rect.united(item.boundingRect())
        self.scene().setSceneRect(rect)
        self.fitInView(self.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

    def zoomStart(self) -> None:
        """Start zoom interactions."""
        self.setInteractionFlag("zoom", True)
