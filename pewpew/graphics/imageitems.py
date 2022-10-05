from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np

from pewlib.process.calc import normalise

from pewpew.actions import qAction
from pewpew.lib.numpyqt import array_to_image, array_to_polygonf

from typing import Any, Optional, List


class ScaledImageItem(QtWidgets.QGraphicsObject):
    """Item to draw image to a defined rect.

    Images scan be bicubic smoothed using `smooth`.
    If `snap` is used, then the 'ItemSendsGeometryChanges' flag must be set.

    Args:
        image: image
        rect: extent of image
        smooth: smooth image using 2x scaling
        snap: snap image position to pixel size
        parent: parent item
    """

    def __init__(
        self,
        image: QtGui.QImage,
        rect: QtCore.QRectF,
        smooth: bool = False,
        snap: bool = True,
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
            self.image_scale = 2
        else:
            self.image = image
            self.image_scale = 1
        self.rect = QtCore.QRectF(rect)  # copy the rect
        self.snap = snap

    def itemChange(
        self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value: Any
    ) -> Any:
        if self.snap and change == QtWidgets.QGraphicsItem.ItemPositionChange:
            pos = QtCore.QPointF(value)
            size = self.pixelSize()
            pos.setX(pos.x() - pos.x() % size.width())
            pos.setY(pos.y() - pos.y() % size.height())
            return pos
        return super().itemChange(change, value)

    def width(self) -> int:
        """Width of image, independant of smoothing."""
        return int(self.image.width() / self.image_scale)

    def height(self) -> int:
        """Height of image, independant of smoothing."""
        return int(self.image.height() / self.image_scale)

    def pixelSize(self) -> QtCore.QSizeF:
        """Size / scaling of an image pixel."""
        return QtCore.QSizeF(
            self.rect.width() / self.width(),
            self.rect.height() / self.height(),
        )

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect

    def mapToData(self, pos: QtCore.QPointF) -> QtCore.QPoint:
        """Map a position to an image pixel coordinate."""
        pixel = self.pixelSize()

        pos -= self.rect.topLeft()
        return QtCore.QPoint(pos.x() / pixel.width(), pos.y() / pixel.height())

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
        colortable: List[int] = None,
        smooth: bool = False,
        parent: QtWidgets.QGraphicsItem = None,
    ) -> "ScaledImageItem":
        """Create a ScaledImageItem from a numpy array.

        Args:
            array: 2d array
            rect: image extent
            colortable: map data using colortable
            smooth: bicubic smoothing
            parent: parent item
        """
        image = array_to_image(array)
        if colortable is not None:
            image.setColorTable(colortable)
            image.setColorCount(len(colortable))
        item = cls(image, rect, smooth, parent=parent)
        return item


class ImageWidgetItem(QtWidgets.QGraphicsObject):
    """Base class for items that act on a ScaledImageItem."""

    def __init__(
        self,
        image: ScaledImageItem,
        data: np.ndarray = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(parent)
        self.image = image
        self.image_data = data

    def imageChanged(self, image: ScaledImageItem, data: np.ndarray) -> None:
        self.image = image
        self.image_data = data


class RulerWidgetItem(ImageWidgetItem):
    """Draws a ruler between two points of a ScaledImageItem.

    Points are selected using the mouse and the length is displayed at the ruler's midpoint.

    Args:
        image: image to measure
        pen: QPen, default to white dashed line
        font: label font
        unit: length unit
        parent: parent item
    """

    def __init__(
        self,
        image: ScaledImageItem,
        pen: QtGui.QPen = None,
        font: QtGui.QFont = None,
        unit: str = "μm",
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(image, None, parent)

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
        self.unit = unit

        self.line = QtCore.QLineF()

    def imageChanged(self, image: ScaledImageItem, data: np.ndarray) -> None:
        pass  # pragma: no cover

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.line.setPoints(event.pos(), event.pos())
            self.text = ""
            self.prepareGeometryChange()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.line.setP2(event.pos())
            self.text = f"{self.line.length():.4g} {self.unit}"
            self.prepareGeometryChange()
        super().mouseMoveEvent(event)

    def boundingRect(self) -> QtCore.QRectF:
        view = next(iter(self.scene().views()))
        angle = self.line.angle()

        fm = QtGui.QFontMetrics(self.font)
        text = fm.boundingRect(self.text)
        text = view.mapToScene(text).boundingRect()

        # Corners above the text height
        n1 = QtCore.QLineF(self.line.p2(), self.line.p1()).normalVector()
        n2 = QtCore.QLineF(self.line.p1(), self.line.p2()).normalVector()
        if 90 < angle < 270:
            n1.setLength(text.height())
            n2.setLength(-text.height())
        else:
            n1.setLength(-text.height())
            n2.setLength(text.height())

        poly = QtGui.QPolygonF([self.line.p1(), n1.p2(), n2.p2(), self.line.p2()])

        w = view.mapToScene(QtCore.QRect(0, 0, 5, 1)).boundingRect().width()
        return poly.boundingRect().marginsAdded(QtCore.QMarginsF(w, w, w, w))

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):
        view = next(iter(self.scene().views()))

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
            width = fm.boundingRect(self.text).width()

            if width < length * 0.9:
                painter.save()
                painter.resetTransform()
                transform = QtGui.QTransform()
                transform.translate(center.x(), center.y())
                transform.rotate(-angle)
                painter.setTransform(transform)
                painter.drawText(-width / 2.0, -fm.descent(), self.text)
                painter.restore()


class ImageSliceWidgetItem(ImageWidgetItem):
    """Draws a 1d data slice between two points of a ScaledImageItem.

    Points are selected using the mouse.
    A context menu option can copy the slice data to the system clipboard.

    Args:
        image: image
        data: image data
        pen: QPen, default to white dotted line
        font: label font
        parent: parent item
    """

    def __init__(
        self,
        image: ScaledImageItem,
        data: np.ndarray,
        pen: QtGui.QPen = None,
        font: QtGui.QFont = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(image, parent)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.white, 2.0, QtCore.Qt.DotLine)
            pen.setCosmetic(True)
            pen.setCapStyle(QtCore.Qt.RoundCap)

        if font is None:
            font = QtGui.QFont()

        self.image_data = data
        self.sliced: Optional[np.ndarray] = None

        self.pen = pen
        self.font = font

        self.line = QtCore.QLineF()
        self.poly = QtGui.QPolygonF()

        self.action_copy_to_clipboard = qAction(
            "insert-text",
            "Copy To Clipboard",
            "Copy slice values to the clipboard.",
            self.actionCopyToClipboard,
        )

    def imageChanged(self, image: ScaledImageItem, data: np.ndarray) -> None:
        super().imageChanged(image, data)
        self.createSlicePoly()

    def actionCopyToClipboard(self):
        if self.sliced is None:
            return
        html = (
            '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
            "<table>"
        )
        text = ""
        for x in self.sliced:
            html += f"<tr><td>{x:.10g}</td></tr>"
            text += f"{x:.10g}\n"
        html += "</table>"

        mime = QtCore.QMimeData()
        mime.setHtml(html)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def createSlicePoly(self) -> QtGui.QPolygonF:
        def connect_nd(ends):
            d = np.diff(ends, axis=0)[0]
            j = np.argmax(np.abs(d))
            D = d[j]
            aD = np.abs(D)
            return ends[0] + (np.outer(np.arange(aD + 1), d) + (aD >> 1)) // aD

        p1 = self.image.mapToData(self.line.p1())
        p2 = self.image.mapToData(self.line.p2())

        if self.line.dx() < 0.0:
            p1, p2 = p2, p1

        view = next(iter(self.scene().views()))
        height = view.mapToScene(QtCore.QRect(0, 0, 1, 100)).boundingRect().height()

        points = connect_nd([[p1.x(), p1.y()], [p2.x(), p2.y()]])
        if points.size > 3:
            self.sliced = self.image_data[points[:, 1], points[:, 0]]

            xs = np.linspace(0.0, self.line.length(), self.sliced.size)
            try:
                ys = -1.0 * normalise(self.sliced, 0.0, height)
            except ValueError:
                self.sliced = None
                self.poly.clear()
                return

            poly = array_to_polygonf(np.stack((xs, ys), axis=1))

            angle = self.line.angle()
            if 90 < angle < 270:
                angle -= 180

            transform = QtGui.QTransform()
            if self.line.dx() < 0.0:
                transform.translate(self.line.p2().x(), self.line.p2().y())
            else:
                transform.translate(self.line.p1().x(), self.line.p1().y())
            transform.rotate(-angle)

            self.poly = transform.map(poly)

    def boundingRect(self) -> QtCore.QRectF:
        view = next(iter(self.scene().views()))
        p1r = view.mapToScene(QtCore.QRect(0, 0, 10, 10)).boundingRect()
        p1r.moveCenter(self.line.p1())
        p2r = p1r.translated(self.line.dx(), self.line.dy())
        return self.poly.boundingRect().united(p1r).united(p2r)

    def shape(self) -> QtGui.QPainterPath:
        p1, p2 = self.line.p1(), self.line.p2()
        if self.line.dx() < 0.0:
            p1, p2 = p2, p1
        path = QtGui.QPainterPath(p1)
        path.lineTo(self.poly.first())
        path.addPolygon(self.poly)
        path.lineTo(p2)
        path.lineTo(p1)
        path.closeSubpath()
        return path

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):

        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setFont(self.font)
        painter.setPen(self.pen)

        painter.drawLine(self.line)
        p1, p2 = self.line.p1(), self.line.p2()

        if self.poly.size() > 8:
            if self.line.dx() < 0.0:
                p1, p2 = p2, p1
            painter.drawLine(p1, self.poly.first())
            painter.drawLine(p2, self.poly.last())

            pen = QtGui.QPen(self.pen)
            pen.setStyle(QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawPolyline(self.poly)

        if not self.line.p1().isNull():
            pen = QtGui.QPen(self.pen)
            pen.setWidth(10)
            painter.setPen(pen)
            painter.drawPoints([self.line.p1(), self.line.p2()])

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()
        menu.addAction(self.action_copy_to_clipboard)
        menu.exec_(event.screenPos())

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            if (
                self.image.rect.left() < event.pos().x() < self.image.rect.right()
                and self.image.rect.top() < event.pos().y() < self.image.rect.bottom()
            ):
                self.line.setPoints(event.pos(), event.pos())
            self.sliced = None
            self.poly = QtGui.QPolygonF()
            self.prepareGeometryChange()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            pos = self.line.p2()
            if self.image.rect.left() < event.pos().x() < self.image.rect.right():
                pos.setX(event.pos().x())
            if self.image.rect.top() < event.pos().y() < self.image.rect.bottom():
                pos.setY(event.pos().y())
            self.line.setP2(pos)
            self.createSlicePoly()
            self.prepareGeometryChange()
        super().mouseMoveEvent(event)
