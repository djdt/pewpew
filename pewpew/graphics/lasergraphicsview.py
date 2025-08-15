import numpy as np
from pewlib.process.register import fft_register_offset
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.graphics.imageitems import LaserImageItem, SnapImageItem
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.overlaygraphics import OverlayGraphicsView
from pewpew.graphics.overlayitems import MetricScaleBarOverlay
from pewpew.graphics.selectionitems import (
    LassoImageSelectionItem,
    RectImageSelectionItem,
    SnapImageSelectionItem,
)
from pewpew.graphics.transformitems import (
    AffineTransformItem,
    ScaleRotateTransformItem,
    TransformItem,
)
from pewpew.graphics.widgetitems import (
    ImageSliceWidgetItem,
    RulerWidgetItem,
    WidgetItem,
)


class IgnoreRightButtonScene(QtWidgets.QGraphicsScene):
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            event.accept()
        else:
            super().mousePressEvent(event)


class LaserGraphicsView(OverlayGraphicsView):
    """The pewpew laser view."""

    def __init__(
        self, options: GraphicsOptions, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(
            IgnoreRightButtonScene(QtCore.QRectF(-1e5, -1e5, 2e5, 2e5), parent),
            parent,
        )
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)

        self.options = options
        self.cursors["selection"] = QtCore.Qt.CursorShape.ArrowCursor

        self.scalebar = MetricScaleBarOverlay(
            font=self.options.font, color=self.options.font_color
        )
        self.scalebar.setVisible(self.options.scalebar)

        self.addOverlayItem(self.scalebar)
        self.scalebar.setPos(0, 10)

        self.options.fontOptionsChanged.connect(self.scalebar.requestPaint)
        self.options.visiblityOptionsChanged.connect(self.updateOverlayVisibility)
        self.viewScaleChanged.connect(self.scalebar.requestPaint)

    def laserItems(self) -> list[LaserImageItem]:
        return [
            item
            for item in self.scene().items(
                self.sceneRect(), QtCore.Qt.ItemSelectionMode.IntersectsItemBoundingRect
            )
            if isinstance(item, LaserImageItem)
        ]

    def selectedLaserItems(self) -> list[LaserImageItem]:
        return [
            item
            for item in self.scene().selectedItems()
            if isinstance(item, LaserImageItem)
        ]

    def alignLaserItemsFFT(self) -> None:
        items = self.selectedLaserItems()
        if len(items) < 2:
            items = self.laserItems()

        base = items[0]
        for item in items[1:]:
            offset = fft_register_offset(base.rawData(), item.rawData())
            psize = item.pixelSize()
            item.setPos(
                base.pos()
                + QtCore.QPointF(offset[1] * psize.width(), offset[0] * psize.height())
            )

    def alignLaserItemsLeftToRight(self) -> None:
        items = self.selectedLaserItems()
        if len(items) < 2:
            items = self.laserItems()
        items = sorted(items, key=lambda item: item.pos().x())

        base = items[0]
        pos = base.pos() + QtCore.QPointF(base.boundingRect().width(), 0.0)
        for item in items[1:]:
            item.setPos(pos)
            pos.setX(pos.x() + item.boundingRect().width())

    def alignLaserItemsTopToBottom(self) -> None:
        items = self.selectedLaserItems()
        if len(items) < 2:
            items = self.laserItems()
        items = sorted(items, key=lambda item: item.pos().y())

        base = items[0]
        pos = base.pos() + QtCore.QPointF(0.0, base.boundingRect().height())
        for item in items[1:]:
            item.setPos(pos)
            pos.setY(pos.y() + item.boundingRect().height())

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self.clearFocus()
        event.accept()

    def startLassoSelection(self) -> None:
        """Select image pixels using a lasso."""
        for item in self.items():
            if isinstance(item, SnapImageSelectionItem):
                self.scene().removeItem(item)

        selection_item = LassoImageSelectionItem(
            allowed_item_types=LaserImageItem, parent=None
        )
        self.scene().addItem(selection_item)
        selection_item.grabMouse()
        self.setInteractionFlag("selection")

    def startRectangleSelection(self) -> None:
        """Select image pixels using a rectangle."""
        for item in self.items():
            if isinstance(item, SnapImageSelectionItem):
                self.scene().removeItem(item)

        selection_item = RectImageSelectionItem(
            allowed_item_types=LaserImageItem, parent=None
        )
        self.scene().addItem(selection_item)
        selection_item.grabMouse()
        self.setInteractionFlag("selection")

    def endSelection(self) -> None:
        """End selection and remove highlight."""
        for item in self.items():
            if isinstance(item, SnapImageSelectionItem):
                self.scene().removeItem(item)
            elif isinstance(item, SnapImageItem):
                item.select(np.zeros([], dtype=bool), [])

        self.setInteractionFlag("selection", False)

    def startRulerWidget(self) -> None:
        """Measure distances using a ruler."""
        for item in self.items():
            if isinstance(item, RulerWidgetItem):
                self.scene().removeItem(item)

        widget = RulerWidgetItem(font=self.options.font)
        widget.setZValue(100.0)
        self.scene().addItem(widget)
        widget.grabMouse()
        self.setInteractionFlag("widget")

    def startSliceWidget(self) -> None:
        """Display 1d slices in image."""
        for item in self.items():
            if isinstance(item, ImageSliceWidgetItem):
                self.scene().removeItem(item)

        widget = ImageSliceWidgetItem(font=self.options.font)
        widget.setZValue(100.0)
        self.scene().addItem(widget)
        widget.grabMouse()
        self.setInteractionFlag("widget")

    def endWidget(self) -> None:
        """End and remove any widgets."""
        for item in self.items():
            if isinstance(item, WidgetItem):
                self.scene().removeItem(item)
        self.setInteractionFlag("widget", False)

    def startTransformAffine(self, item: SnapImageItem | None = None) -> None:
        if item is None:
            item = self.scene().focusItem()
        if not isinstance(item, SnapImageItem):
            return

        widget = AffineTransformItem(item)
        self.scene().addItem(widget)
        widget.grabMouse()
        self.setInteractionFlag("transform")

    def startTransformScale(self, item: SnapImageItem | None = None) -> None:
        if item is None:
            item = self.scene().focusItem()
        if not isinstance(item, SnapImageItem):
            return

        widget = ScaleRotateTransformItem(item)
        self.scene().addItem(widget)
        widget.grabMouse()
        self.setInteractionFlag("transform")

    def resetTransform(self, item: SnapImageItem | None = None) -> None:
        self.endTransform()
        if item is None:
            item = self.scene().focusItem()
        if not isinstance(item, SnapImageItem):
            return
        item.resetTransform()

    def endTransform(self) -> None:
        for item in self.items():
            if isinstance(item, TransformItem):
                self.scene().removeItem(item)
        self.setInteractionFlag("transform", False)

    def updateOverlayVisibility(self) -> None:
        self.scalebar.setVisible(self.options.scalebar)
        self.scalebar.requestPaint()

    def zoomReset(self) -> None:
        # Compute a reasonable estimate of the bounding rect
        rect = QtCore.QRectF()
        for item in self.items():
            if isinstance(item, SnapImageItem) and item.isVisible():
                rect = rect.united(item.sceneBoundingRect())
        self.fitInView(
            rect.marginsAdded(QtCore.QMarginsF(100, 100, 100, 100)),
            QtCore.Qt.KeepAspectRatio,
        )

        # Get the actual bounding rect
        super().zoomReset()
