from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np

from pewpew.graphics.imageitems import ScaledImageItem
from pewpew.graphics.util import polygonf_contains_points

from pewpew.lib.numpyqt import polygonf_to_array

from typing import Dict, Generator, Optional


class SelectionItem(QtWidgets.QGraphicsObject):
    selectionChanged = QtCore.Signal(np.ndarray, "QStringList")

    def __init__(
        self,
        modes: Optional[Dict[QtCore.Qt.KeyboardModifier, str]] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self.setZValue(99)

        self.modes = modes or {}

    def modifierModes(
        self, modifiers: QtCore.Qt.KeyboardModifier
    ) -> Generator[str, None, None]:
        for k, v in self.modes.items():
            if k & modifiers:
                yield v


class ScaledImageSelectionItem(SelectionItem):
    """Base class for image selection items.

    Args:
        image: image to select from
        modes: map keyboard modifers to modes,
            default maps shift to 'add', control to 'subtract'
        parent: parent item
    """

    def __init__(
        self,
        image: Optional[ScaledImageItem] = None,
        modes: Optional[Dict[QtCore.Qt.KeyboardModifier, str]] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        _modes = {QtCore.Qt.ShiftModifier: "add", QtCore.Qt.ControlModifier: "subtract"}
        if modes is not None:
            _modes.update(modes)
        super().__init__(modes=_modes, parent=parent)

        self.rect = QtCore.QRectF(image.rect)
        self.image_shape = (image.height(), image.width())

    def pixelSize(self) -> QtCore.QSizeF:
        return QtCore.QSizeF(
            self.rect.width() / self.image_shape[1],
            self.rect.height() / self.image_shape[0],
        )

    def snapPos(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        pixel = self.pixelSize()
        x = round(pos.x() / pixel.width()) * pixel.width()
        y = round(pos.y() / pixel.height()) * pixel.height()
        return QtCore.QPointF(x, y)


class LassoImageSelectionItem(ScaledImageSelectionItem):
    """Selection using a lasso that follows mouse movements."""

    def __init__(
        self,
        image: ScaledImageItem,
        modes: Optional[Dict[QtCore.Qt.Modifier, str]] = None,
        pen: Optional[QtGui.QPen] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(image, modes=modes, parent=parent)

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
        if not event.button() & QtCore.Qt.LeftButton:
            return
        self.poly.clear()
        self.poly.append(self.snapPos(event.pos()))

        self.prepareGeometryChange()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.buttons() & QtCore.Qt.LeftButton:
            return
        if self.poly.size() == 0:
            return
        pos = self.snapPos(event.pos())
        if self.poly.last() != pos:
            self.poly.append(pos)
            self.prepareGeometryChange()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.button() & QtCore.Qt.LeftButton:
            return
        modes = list(self.modifierModes(event.modifiers()))
        pixel = self.pixelSize()

        array = polygonf_to_array(self.poly)
        # Get start and end points of area
        x1, x2 = np.amin(array[:, 0]), np.amax(array[:, 0])
        y1, y2 = np.amin(array[:, 1]), np.amax(array[:, 1])
        # Bound to image area
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2 = min(x2, self.rect.width() - pixel.width() / 2.0)
        y2 = min(y2, self.rect.height() - pixel.height() / 2.0)
        # Generate pixel centers
        xs = np.arange(x1, x2, pixel.width()) + pixel.width() / 2.0
        ys = np.arange(y1, y2, pixel.height()) + pixel.height() / 2.0
        X, Y = np.meshgrid(xs, ys)
        pixels = np.stack((X.flat, Y.flat), axis=1)

        # Get mask of selected area
        mask = np.zeros(self.image_shape, dtype=bool)
        polymask = polygonf_contains_points(self.poly, pixels).reshape(ys.size, xs.size)
        # Insert
        ix, iy = int(x1 / pixel.width()), int(y1 / pixel.height())
        mask[iy : iy + ys.size, ix : ix + xs.size] = polymask

        # self.poly.append(self.poly.first())
        self.poly.clear()
        self.prepareGeometryChange()

        self.selectionChanged.emit(mask, modes)

    def paint(
        self,
        painter: QtGui.QPainter,
        options: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        painter.save()

        painter.setPen(self.pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.drawPolyline(self.poly)

        painter.restore()


class RectImageSelectionItem(ScaledImageSelectionItem):
    """Selection using a rectangle."""

    def __init__(
        self,
        image: ScaledImageItem,
        modes: Optional[Dict[QtCore.Qt.Modifier, str]] = None,
        pen: Optional[QtGui.QPen] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(image, modes=modes, parent=parent)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.white, 2.0)

        self.pen = pen
        self.pen.setCosmetic(True)

        self._rect = QtCore.QRectF()

    def boundingRect(self) -> QtCore.QRectF:
        return self._rect.normalized()

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.button() & QtCore.Qt.LeftButton:
            return
        self._rect.setTopLeft(event.pos())
        self._rect.setBottomRight(event.pos())
        self.prepareGeometryChange()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.buttons() & QtCore.Qt.LeftButton:
            return
        self._rect.setBottomRight(event.pos())
        self.prepareGeometryChange()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.button() & QtCore.Qt.LeftButton:
            return
        modes = list(self.modifierModes(event.modifiers()))

        px, py = (
            self.rect.width() / self.image_shape[1],
            self.rect.height() / self.image_shape[0],
        )  # pixel size

        x1, y1, x2, y2 = self._rect.normalized().getCoords()
        x1 = np.round(x1 / px).astype(int)
        x2 = np.round(x2 / px).astype(int)
        y1 = np.round(y1 / py).astype(int)
        y2 = np.round(y2 / py).astype(int)

        mask = np.zeros(self.image_shape, dtype=bool)
        mask[y1:y2, x1:x2] = True

        self._rect = QtCore.QRectF()
        self.prepareGeometryChange()

        self.selectionChanged.emit(mask, modes)

    def paint(
        self,
        painter: QtGui.QPainter,
        options: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        painter.save()

        painter.setPen(self.pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.drawRect(self._rect)

        painter.restore()
