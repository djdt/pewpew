from typing import Dict, Generator, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.graphics.imageitems import SnapImageItem
from pewpew.graphics.util import polygonf_contains_points
from pewpew.lib.numpyqt import polygonf_to_array


class SelectionItem(QtWidgets.QGraphicsObject):
    selectionChanged = QtCore.Signal(np.ndarray, "QStringList")

    def __init__(
        self,
        modes: dict[QtCore.Qt.KeyboardModifier, str] | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(parent)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self.setZValue(100.0)

        self.modes = modes or {}

    def modifierModes(
        self, modifiers: QtCore.Qt.KeyboardModifier
    ) -> Generator[str, None, None]:
        for k, v in self.modes.items():
            if k & modifiers:
                yield v


class SnapImageSelectionItem(SelectionItem):
    """Base class for image selection items.

    Args:
        image: image to select from
        modes: map keyboard modifers to modes,
            default maps shift to 'add', control to 'subtract'
        parent: parent item
    """

    def __init__(
        self,
        modes: dict[QtCore.Qt.KeyboardModifier, str] | None = None,
        allowed_item_types: tuple[type] | type = SnapImageItem,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        _modes = {QtCore.Qt.ShiftModifier: "add", QtCore.Qt.ControlModifier: "subtract"}
        if modes is not None:
            _modes.update(modes)
        super().__init__(modes=_modes, parent=parent)

        self.item: SnapImageItem | None = None
        self.allowed_item_types = allowed_item_types

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        try:
            self.item = next(
                item
                for item in self.scene().items(
                    event.scenePos(), QtCore.Qt.IntersectsItemBoundingRect
                )
                if isinstance(item, self.allowed_item_types)
                and item.acceptedMouseButtons() & event.button()
            )
        except StopIteration:
            self.item = None

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.item = None


class LassoImageSelectionItem(SnapImageSelectionItem):
    """Selection using a lasso that follows mouse movements."""

    def __init__(
        self,
        modes: dict[QtCore.Qt.KeyboardModifier, str] | None = None,
        pen: QtGui.QPen | None = None,
        allowed_item_types: tuple[type] | type = SnapImageItem,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(
            modes=modes, allowed_item_types=allowed_item_types, parent=parent
        )

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.white, 2.0)

        self.pen = pen
        self.pen.setCosmetic(True)

        self.poly = QtGui.QPolygonF()

    def boundingRect(self) -> QtCore.QRectF:
        return self.poly.boundingRect()

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addPolygon(self.poly)
        return path

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        super().mousePressEvent(event)
        if not event.button() & QtCore.Qt.LeftButton or self.item is None:
            return

        self.poly.clear()
        self.poly.append(self.item.snapPos(event.pos()))

        self.prepareGeometryChange()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.buttons() & QtCore.Qt.LeftButton or self.item is None:
            return
        if self.poly.size() == 0:
            return

        pos = self.item.snapPos(event.pos())
        if self.poly.last() != pos:
            self.poly.append(pos)
            self.prepareGeometryChange()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.button() & QtCore.Qt.LeftButton or self.item is None:
            return

        modes = list(self.modifierModes(event.modifiers()))
        pixel = self.item.pixelSize()
        rect = self.item.boundingRect()
        size = self.item.imageSize()

        poly = self.mapToItem(self.item, self.poly)

        array = polygonf_to_array(poly)
        # Get start and end points of area
        x1, x2 = np.amin(array[:, 0]), np.amax(array[:, 0])
        y1, y2 = np.amin(array[:, 1]), np.amax(array[:, 1])
        # Bound to image area
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2 = min(x2, rect.width() - pixel.width() / 2.0)
        y2 = min(y2, rect.height() - pixel.height() / 2.0)
        # Generate pixel centers
        xs = np.arange(x1, x2, pixel.width()) + pixel.width() / 2.0
        ys = np.arange(y1, y2, pixel.height()) + pixel.height() / 2.0
        X, Y = np.meshgrid(xs, ys)
        pixels = np.stack((X.flat, Y.flat), axis=1)

        # Get mask of selected area
        mask = np.zeros((size.height(), size.width()), dtype=bool)
        polymask = polygonf_contains_points(poly, pixels).reshape(ys.size, xs.size)
        # Insert
        ix, iy = int(x1 / pixel.width()), int(y1 / pixel.height())
        mask[iy : iy + ys.size, ix : ix + xs.size] = polymask

        self.poly.clear()
        self.prepareGeometryChange()

        self.item.select(mask, modes)
        super().mouseReleaseEvent(event)

    def paint(
        self,
        painter: QtGui.QPainter,
        options: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()

        painter.setPen(self.pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.drawPolyline(self.poly)

        painter.restore()


class RectImageSelectionItem(SnapImageSelectionItem):
    """Selection using a rectangle."""

    def __init__(
        self,
        modes: dict[QtCore.Qt.KeyboardModifier, str] | None = None,
        pen: QtGui.QPen | None = None,
        allowed_item_types: tuple[type] | type = SnapImageItem,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(
            modes=modes, allowed_item_types=allowed_item_types, parent=parent
        )

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.white, 2.0)

        self.pen = pen
        self.pen.setCosmetic(True)

        self._rect = QtCore.QRectF()

    def boundingRect(self) -> QtCore.QRectF:
        return self._rect.normalized()

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        super().mousePressEvent(event)
        if not event.button() & QtCore.Qt.LeftButton or self.item is None:
            return

        pos = self.item.snapPos(event.pos())

        self._rect.setTopLeft(pos)
        self._rect.setBottomRight(pos)
        self.prepareGeometryChange()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.buttons() & QtCore.Qt.LeftButton or self.item is None:
            return

        pos = self.item.snapPos(event.pos())
        self._rect.setBottomRight(pos)
        self.prepareGeometryChange()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.button() & QtCore.Qt.LeftButton or self.item is None:
            return
        modes = list(self.modifierModes(event.modifiers()))

        rect = self.item.boundingRect()
        size = self.item.imageSize()

        px, py = (
            rect.width() / size.width(),
            rect.height() / size.height(),
        )  # pixel size

        x1, y1, x2, y2 = (
            self.mapToItem(self.item, self._rect.normalized())
            .boundingRect()
            .getCoords()
        )
        x1 = np.round(x1 / px).astype(int)
        x2 = np.round(x2 / px).astype(int)
        y1 = np.round(y1 / py).astype(int)
        y2 = np.round(y2 / py).astype(int)

        mask = np.zeros((size.height(), size.width()), dtype=bool)
        mask[y1:y2, x1:x2] = True

        self._rect = QtCore.QRectF()
        self.prepareGeometryChange()

        self.item.select(mask, modes)
        super().mouseReleaseEvent(event)

    def paint(
        self,
        painter: QtGui.QPainter,
        options: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()

        painter.setPen(self.pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.drawRect(self._rect)

        painter.restore()
