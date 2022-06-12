import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.graphics.imageitems import SnapImageItem
from pewpew.graphics.selectionitems import (
    LassoImageSelectionItem,
    RectImageSelectionItem,
    SnapImageSelectionItem,
)
from pewpew.graphics.widgetitems import (
    ImageSliceWidgetItem,
    RulerWidgetItem,
    WidgetItem,
)
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.overlaygraphics import OverlayGraphicsView
from pewpew.graphics.overlayitems import (
    MetricScaleBarOverlay,
)

from typing import Optional


class LaserGraphicsView(OverlayGraphicsView):
    """The pewpew laser view.

    Displays the image with correct scaling and an overlay label, sizebar and colorbar.
    If a selection is made the 'mask' is updated and a highlight is applied to sselected pixels.
    """

    def __init__(
        self, options: GraphicsOptions, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(
            QtWidgets.QGraphicsScene(QtCore.QRectF(0, 0, 1000, 1000), parent), parent
        )

        self.options = options
        self.cursors["selection"] = QtCore.Qt.ArrowCursor

        self.scalebar = MetricScaleBarOverlay(
            font=self.options.font, color=self.options.font_color
        )
        self.scalebar.setVisible(self.options.scalebar)

        self.addOverlayItem(self.scalebar)
        self.scalebar.setPos(0, 10)

        self.options.fontOptionsChanged.connect(self.scalebar.requestPaint)
        self.options.visiblityOptionsChanged.connect(self.updateOverlayVisibility)
        self.viewScaleChanged.connect(self.scalebar.requestPaint)


    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self.clearFocus()
        event.accept()

    def startLassoSelection(self) -> None:
        """Select image pixels using a lasso."""
        for item in self.items():
            if isinstance(item, SnapImageSelectionItem):
                self.scene().removeItem(item)

        selection_item = LassoImageSelectionItem(parent=None)
        self.scene().addItem(selection_item)
        selection_item.grabMouse()
        self.setInteractionFlag("selection")

    def startRectangleSelection(self) -> None:
        """Select image pixels using a rectangle."""
        for item in self.items():
            if isinstance(item, SnapImageSelectionItem):
                self.scene().removeItem(item)

        selection_item = RectImageSelectionItem(parent=None)
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

    def updateOverlayVisibility(self) -> None:
        self.scalebar.setVisible(self.options.scalebar)
        self.scalebar.requestPaint()

    def zoomReset(self) -> None:
        # Compute a reasonable estimate of the bounding rect
        rect = QtCore.QRectF()
        for item in self.items():
            if isinstance(item, SnapImageItem) and item.isVisible():
                rect = rect.united(item.sceneBoundingRect())
        self.fitInView(rect.marginsAdded(QtCore.QMarginsF(50, 50, 50, 50)), QtCore.Qt.KeepAspectRatio)

        # Get the actual bounding rect
        rect = self.itemsBoundingRect()
        self.scene().setSceneRect(rect)
        self.fitInView(rect, QtCore.Qt.KeepAspectRatio)
