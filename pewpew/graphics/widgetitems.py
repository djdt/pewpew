from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np

from pewlib.process.calc import normalise

from pewpew.actions import qAction
from pewpew.lib.numpyqt import array_to_polygonf

from pewpew.graphics.imageitems import SnapImageItem

from typing import Optional


class WidgetItem(QtWidgets.QGraphicsObject):
    pass


class SnapImageWidgetItem(WidgetItem):
    def __init__(
        self,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent=parent)
        self.item: Optional[SnapImageItem] = None

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        try:
            self.item = next(
                item
                for item in self.scene().items(
                    event.scenePos(), QtCore.Qt.IntersectsItemBoundingRect
                )
                if isinstance(item, SnapImageItem)
                and item.acceptedMouseButtons() & event.button()
            )
        except StopIteration:
            self.item = None

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.item = None


class RulerWidgetItem(WidgetItem):
    """Draws a ruler between two points in a scene.

    Points are selected using the mouse and the length is displayed at the ruler's midpoint.

    Args:
        pen: QPen, default to white dashed line
        font: label font
        unit: length unit
        parent: parent item
    """

    def __init__(
        self,
        pen: Optional[QtGui.QPen] = None,
        font: Optional[QtGui.QFont] = None,
        unit: str = "μm",
        parent: Optional[QtWidgets.QGraphicsItem] = None,
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
        self.unit = unit

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
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        view = next(iter(self.scene().views()))

        painter.save()
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
                painter.resetTransform()
                transform = QtGui.QTransform()
                transform.translate(center.x(), center.y())
                transform.rotate(-angle)
                painter.setTransform(transform)
                painter.drawText(-width / 2.0, -fm.descent(), self.text)
        painter.restore()


class ImageSliceWidgetItem(WidgetItem):
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
        pen: Optional[QtGui.QPen] = None,
        font: Optional[QtGui.QFont] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.white, 2.0, QtCore.Qt.DotLine)
            pen.setCosmetic(True)
            pen.setCapStyle(QtCore.Qt.RoundCap)

        if font is None:
            font = QtGui.QFont()

        self.image: Optional[SnapImageItem] = None
        self.sliced: Optional[np.ndarray] = None

        self.pen = pen
        self.font = font

        self.line = QtCore.QLineF()
        self.poly = QtGui.QPolygonF()

        self.action_copy_to_clipboard = qAction(
            "insert-text",
            "Copy To Clipboard",
            "Copy slice values to the clipboard.",
            self.copyToClipboard,
        )

    def itemImageChanged(self) -> None:
        self.createSlicePoly()

    def copyToClipboard(self):
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

    def createSlicePoly(self) -> None:
        # All pos in scene coordinates
        def connect_nd(ends):
            d = np.diff(ends, axis=0)[0]
            j = np.argmax(np.abs(d))
            D = d[j]
            aD = np.abs(D)
            return ends[0] + (np.outer(np.arange(aD + 1), d) + (aD >> 1)) // aD

        p1 = self.item.mapToData(self.item.mapFromScene(self.line.p1()))
        p2 = self.item.mapToData(self.item.mapFromScene(self.line.p2()))

        if self.line.dx() < 0.0:
            p1, p2 = p2, p1

        view = next(iter(self.scene().views()))
        height = view.mapToScene(QtCore.QRect(0, 0, 1, 100)).boundingRect().height()

        points = connect_nd([[p1.x(), p1.y()], [p2.x(), p2.y()]])
        if points.size > 3:
            self.sliced = self.item.rawData()[points[:, 1], points[:, 0]]
            if self.sliced.ndim > 1:  # for RGB and other images
                self.sliced = np.mean(self.sliced, axis=1)

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
        widget: Optional[QtWidgets.QWidget] = None,
    ):

        painter.save()
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
        painter.restore()

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()
        menu.addAction(self.action_copy_to_clipboard)
        menu.exec_(event.screenPos())

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        # All pos in scene coordinates
        if not event.buttons() & QtCore.Qt.LeftButton:
            return

        item = self.scene().itemAt(event.scenePos(), QtGui.QTransform())
        if isinstance(item, SnapImageItem):
            self.item = item
            self.item.imageChanged.connect(self.itemImageChanged)

            self.line.setPoints(event.pos(), event.pos())
            self.sliced = None
            self.poly = QtGui.QPolygonF()
            self.prepareGeometryChange()
        else:
            self.item = None

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        # All pos in scene coordinates
        if event.buttons() & QtCore.Qt.LeftButton and self.item is not None:
            pos = self.line.p2()
            rect = self.item.sceneBoundingRect()
            if rect.left() < event.pos().x() < rect.right():
                pos.setX(event.pos().x())
            if rect.top() < event.pos().y() < rect.bottom():
                pos.setY(event.pos().y())
            self.line.setP2(pos)
            self.createSlicePoly()
            self.prepareGeometryChange()
        super().mouseMoveEvent(event)
