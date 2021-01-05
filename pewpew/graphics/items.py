from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np

from pewpew.graphics.util import polygon_contains_points

from pewpew.lib.numpyqt import array_to_image

from typing import Dict, Generator, List


class ResizeableRectItem(QtWidgets.QGraphicsRectItem):
    cursors = {
        "left": QtCore.Qt.SizeHorCursor,
        "right": QtCore.Qt.SizeHorCursor,
        "top": QtCore.Qt.SizeVerCursor,
        "bottom": QtCore.Qt.SizeVerCursor,
    }

    def __init__(
        self,
        rect: QtCore.QRectF,
        selection_dist: int,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(rect, parent)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self.selection_dist = selection_dist
        self.selectedEdge: str = None

    def edgeAt(self, pos: QtCore.QPointF) -> str:
        view = next(iter(self.scene().views()), None)
        if view is None:
            return None
        dist = (
            view.mapToScene(QtCore.QRect(0, 0, self.selection_dist, 1))
            .boundingRect()
            .width()
        )

        if pos.x() < self.rect().left() + dist:
            return "left"
        elif pos.x() > self.rect().right() - dist:
            return "right"
        elif pos.y() < self.rect().top() + dist:
            return "top"
        elif pos.y() > self.rect().bottom() - dist:
            return "bottom"
        else:
            return None

    def rectChanged(self) -> None:
        pass

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if self.isSelected():
            edge = self.edgeAt(event.pos())
            if edge in self.cursors:
                self.setCursor(self.cursors[edge])
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.setCursor(QtCore.Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.isSelected():
            self.selectedEdge = self.edgeAt(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.selectedEdge is None:
            super().mouseMoveEvent(event)
        else:
            rect = self.rect()
            if self.selectedEdge == "left" and event.pos().x() < rect.right():
                rect.setLeft(event.pos().x())
            elif self.selectedEdge == "right" and event.pos().x() > rect.left():
                rect.setRight(event.pos().x())
            elif self.selectedEdge == "top" and event.pos().y() < rect.bottom():
                rect.setTop(event.pos().y())
            elif self.selectedEdge == "bottom" and event.pos().y() > rect.top():
                rect.setBottom(event.pos().y())

            self.prepareGeometryChange()
            self.setRect(rect)
            self.rectChanged()


class ScaledImageItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        image: QtGui.QImage,
        rect: QtCore.QRectF,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(parent)
        self.setCacheMode(
            QtWidgets.QGraphicsItem.DeviceCoordinateCache
        )  # Speed up redraw of image
        self.image = image
        self.rect = QtCore.QRectF(rect)  # copy the rect

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):
        painter.drawImage(self.rect, self.image)

    @classmethod
    def fromArray(
        cls,
        array: np.ndarray,
        rect: QtCore.QRectF,
        colortable: np.ndarray = None,
        parent: QtWidgets.QGraphicsItem = None,
    ) -> "ScaledImageItem":
        image = array_to_image(array)
        if colortable is not None:
            image.setColorTable(colortable)
            image.setColorCount(len(colortable))
        item = cls(image, rect, parent=parent)
        return item


class SelectionItem(QtWidgets.QGraphicsObject):
    selectionChanged = QtCore.Signal()

    def __init__(
        self,
        modes: Dict[QtCore.Qt.KeyboardModifier, str] = None,
        parent: QtWidgets.QGraphicsItem = None,
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
    def __init__(
        self,
        image: ScaledImageItem = None,
        modes: Dict[QtCore.Qt.KeyboardModifier, str] = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        _modes = {QtCore.Qt.ShiftModifier: "add", QtCore.Qt.ControlModifier: "subtract"}
        if modes is not None:
            _modes.update(modes)
        super().__init__(modes=_modes, parent=parent)

        self.rect = QtCore.QRectF(image.rect)
        self.size = (image.image.width(), image.image.height())

        self.mask = np.zeros(self.size, dtype=np.bool)

    def maskAsImage(self, color: QtGui.QColor = None) -> ScaledImageItem:
        if color is None:
            color = QtGui.QColor(QtCore.Qt.white)
            color.setAlpha(255)
        data = self.mask.astype(np.uint8)
        item = ScaledImageItem.fromArray(
            data,
            self.rect,
            colortable=[0, color.rgba()],
        )
        return item

    def updateMask(self, mask: np.ndarray, modes: List[str]) -> None:
        mask = mask.astype(np.bool)
        if "add" in modes:
            self.mask = np.logical_or(self.mask, mask)
        elif "subtract" in modes:
            self.mask = np.logical_and(self.mask, ~mask)
        elif "intersect" in modes:
            self.mask = np.logical_and(self.mask, mask)
        elif "difference" in modes:
            self.mask = np.logical_xor(self.mask, mask)
        else:
            self.mask = mask

    def pixelPositions(self, shift: float = 0.5) -> np.ndarray:
        x, y, w, h = self.rect.getRect()
        px, py = w / self.size[0], h / self.size[1]  # pixel size

        xs = np.linspace(x, x + w, self.size[0], endpoint=False) + (px * shift)
        ys = np.linspace(y, y + h, self.size[1], endpoint=False) + (py * shift)
        X, Y = np.meshgrid(xs, ys)
        return np.stack((X.flat, Y.flat), axis=1)


class LassoImageSelectionItem(ScaledImageSelectionItem):
    def __init__(
        self,
        image: ScaledImageItem,
        modes: Dict[QtCore.Qt.Modifier, str] = None,
        pen: QtGui.QPen = None,
        minimum_pixel_distance: int = 5,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(image, parent=parent)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.white, 2.0)

        self.pen = pen
        self.pen.setCosmetic(True)
        self.pixel_distance = minimum_pixel_distance

        self.poly = QtGui.QPolygonF()
        self._last_sceen_pos: QtCore.QPoint = None

    def boundingRect(self) -> QtCore.QRectF:
        return self.poly.boundingRect()

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addPolygon(self.poly)
        return path

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.poly.clear()
        self.poly.append(event.pos())

        self._last_sceen_pos = event.screenPos()
        self.prepareGeometryChange()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.buttons() & QtCore.Qt.LeftButton:
            return
        if self.poly.size() == 0:
            return
        if (
            QtCore.QLineF(self._last_sceen_pos, event.screenPos()).length()
            > self.pixel_distance
        ):
            self.poly.append(event.pos())
            self._last_sceen_pos = event.screenPos()
            self.prepareGeometryChange()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        modes = list(self.modifierModes(event.modifiers()))

        pixels = self.pixelPositions()
        mask = polygon_contains_points(self.poly, pixels).reshape(
            self.size[1], self.size[0]
        )
        self.updateMask(mask, modes)

        self.poly.clear()
        self.prepareGeometryChange()

        self.selectionChanged.emit()

    def paint(
        self,
        painter: QtGui.QPainter,
        options: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):
        painter.save()

        painter.setPen(self.pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.drawPolyline(self.poly)

        painter.restore()


class RectImageSelectionItem(ScaledImageSelectionItem):
    def __init__(
        self,
        image: ScaledImageItem,
        modes: Dict[QtCore.Qt.Modifier, str] = None,
        pen: QtGui.QPen = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(image, parent=parent)

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
            self.rect.width() / self.size[0],
            self.rect.height() / self.size[1],
        )  # pixel size

        x1, y1, x2, y2 = self._rect.normalized().getCoords()
        x1 = np.round(x1 / px).astype(int)
        x2 = np.round(x2 / px).astype(int)
        y1 = np.round(y1 / py).astype(int)
        y2 = np.round(y2 / py).astype(int)

        mask = np.zeros((self.size[1], self.size[0]), dtype=np.bool)
        mask[y1:y2, x1:x2] = True

        self.updateMask(mask, modes)

        self._rect = QtCore.QRectF()
        self.prepareGeometryChange()

        self.selectionChanged.emit()

    def paint(
        self,
        painter: QtGui.QPainter,
        options: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):
        painter.save()

        painter.setPen(self.pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.drawRect(self._rect)

        painter.restore()
