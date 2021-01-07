from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np

from pewpew.graphics.util import polygonf_contains_points

from pewpew.lib.numpyqt import array_to_image, polygonf_to_array

from typing import Dict, Generator


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
        self.selected_edge: str = None

    def boundingRect(self) -> QtCore.QRectF:
        rect = super().boundingRect()
        view = next(iter(self.scene().views()), None)
        if view is None:
            return rect

        dist = (
            view.mapToScene(QtCore.QRect(0, 0, self.selection_dist, 1))
            .boundingRect()
            .width()
        )
        return rect.marginsAdded(QtCore.QMarginsF(dist, dist, dist, dist))

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        rect = self.boundingRect()
        path.addRect(rect)
        return path

    def edgeAt(self, pos: QtCore.QPointF) -> str:
        view = next(iter(self.scene().views()), None)
        if view is None:
            return None
        dist = (
            view.mapToScene(QtCore.QRect(0, 0, self.selection_dist, 1))
            .boundingRect()
            .width()
        )

        if abs(self.rect().left() - pos.x()) < dist:
            return "left"
        elif abs(self.rect().right() - pos.x()) < dist:
            return "right"
        elif abs(self.rect().top() - pos.y()) < dist:
            return "top"
        elif abs(self.rect().bottom() - pos.y()) < dist:
            return "bottom"
        else:
            return None

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
            self.selected_edge = self.edgeAt(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        pos = self.itemChange(QtWidgets.QGraphicsItem.ItemPositionChange, event.pos())
        if self.selected_edge is None:
            super().mouseMoveEvent(event)
        else:
            rect = self.rect()
            if self.selected_edge == "left" and pos.x() < rect.right():
                rect.setLeft(pos.x())
            elif self.selected_edge == "right" and pos.x() > rect.left():
                rect.setRight(pos.x())
            elif self.selected_edge == "top" and pos.y() < rect.bottom():
                rect.setTop(pos.y())
            elif self.selected_edge == "bottom" and pos.y() > rect.top():
                rect.setBottom(pos.y())

            self.prepareGeometryChange()
            self.setRect(rect)


class RulerItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        pen: QtGui.QPen = None,
        font: QtGui.QFont = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(parent)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.white, 2.0)
            pen.setCosmetic(True)
            pen.setStyle(QtCore.Qt.DashLine)
            pen.setCapStyle(QtCore.Qt.RoundCap)

        if font is None:
            font = QtGui.QFont()

        self.pen = pen
        self.font = font
        self.text = ""

        self.line = QtCore.QLineF()

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.line.setPoints(event.pos(), event.pos())
            self.text = ""
            self.prepareGeometryChange()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.line.setP2(event.pos())
            self.text = f"{self.line.length():.4g} μm"
            self.prepareGeometryChange()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.line.setP2(event.pos())
            self.text = f"{self.line.length():.4g} μm"
            self.prepareGeometryChange()
        super().mouseReleaseEvent(event)

    def boundingRect(self) -> QtCore.QRectF:
        view = next(iter(self.scene().views()))
        angle = self.line.angle()

        fm = QtGui.QFontMetrics(self.font)
        text = fm.boundingRect(self.text)
        text = view.mapToScene(text).boundingRect()

        if 90 < angle < 270:
            norm = QtCore.QLineF(self.line.center(), self.line.p1()).normalVector()
        else:
            norm = QtCore.QLineF(self.line.center(), self.line.p2()).normalVector()
        norm.setLength(text.height())

        poly = QtGui.QPolygonF([self.line.p1(), norm.p2(), self.line.p2()])

        w = view.mapToScene(QtCore.QRect(0, 0, 5, 1)).boundingRect().width()
        return poly.boundingRect().marginsAdded(QtCore.QMarginsF(w, w, w, w))

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):
        view = next(iter(self.scene().views()), None)

        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setFont(self.font)
        painter.setPen(self.pen)

        fm = painter.fontMetrics()

        painter.drawLine(self.line)

        if not self.line.p1().isNull():
            pen = QtGui.QPen(self.pen)
            pen.setWidth(10)
            painter.setPen(pen)
            painter.drawPoints([self.line.p1(), self.line.p2()])
            painter.setPen(self.pen)

        if view is not None and self.text != "":
            angle = self.line.angle()
            if 90 < angle < 270:
                angle -= 180
            center = view.mapFromScene(self.line.center())
            length = (
                view.mapFromScene(QtCore.QRectF(0, 0, self.line.length(), 1))
                .boundingRect()
                .width()
            )
            width = fm.width(self.text)

            if width < length:
                painter.save()
                painter.resetTransform()
                transform = QtGui.QTransform()
                transform.translate(center.x(), center.y())
                transform.rotate(-angle)
                painter.setTransform(transform)
                painter.drawText(-width / 2.0, -fm.descent(), self.text)
                painter.restore()


class ScaledImageItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        image: QtGui.QImage,
        rect: QtCore.QRectF,
        smooth: bool = False,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(parent)
        self.setCacheMode(
            QtWidgets.QGraphicsItem.DeviceCoordinateCache
        )  # Speed up redraw of image
        if smooth:
            self.image = image.scaledToHeight(
                image.height() * 2, QtCore.Qt.SmoothTransformation
            )
            self.scale = 2
        else:
            self.image = image
            self.scale = 1
        self.rect = QtCore.QRectF(rect)  # copy the rect

    def width(self) -> int:
        return self.image.width() // self.scale

    def height(self) -> int:
        return self.image.height() // self.scale

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
        smooth: bool = False,
        parent: QtWidgets.QGraphicsItem = None,
    ) -> "ScaledImageItem":
        image = array_to_image(array)
        if colortable is not None:
            image.setColorTable(colortable)
            image.setColorCount(len(colortable))
        item = cls(image, rect, smooth, parent=parent)
        return item


class SelectionItem(QtWidgets.QGraphicsObject):
    selectionChanged = QtCore.Signal(np.ndarray, "QStringList")

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

        self.poly = QtGui.QPolygonF()

    def boundingRect(self) -> QtCore.QRectF:
        return self.poly.boundingRect()

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        path.addPolygon(self.poly)
        return path

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
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
        mask = np.zeros(self.image_shape, dtype=np.bool)
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
            self.rect.width() / self.image_shape[1],
            self.rect.height() / self.image_shape[0],
        )  # pixel size

        x1, y1, x2, y2 = self._rect.normalized().getCoords()
        x1 = np.round(x1 / px).astype(int)
        x2 = np.round(x2 / px).astype(int)
        y1 = np.round(y1 / py).astype(int)
        y2 = np.round(y2 / py).astype(int)

        mask = np.zeros(self.image_shape, dtype=np.bool)
        mask[y1:y2, x1:x2] = True

        self._rect = QtCore.QRectF()
        self.prepareGeometryChange()

        self.selectionChanged.emit(mask, modes)

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
