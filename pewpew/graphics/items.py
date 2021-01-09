from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np

from pewlib.process.calc import normalise

from pewpew.actions import qAction

from pewpew.lib.numpyqt import array_to_image, array_to_polygonf


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

            if width < length * 0.9:
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

    def pixelSize(self) -> QtCore.QSizeF:
        return QtCore.QSizeF(
            self.rect.width() / self.image.width(),
            self.rect.height() / self.image.height(),
        )

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect

    def mapToData(self, pos: QtCore.QPointF) -> QtCore.QPoint:
        px = self.rect.width() / self.image.width()
        py = self.rect.height() / self.image.height()

        pos -= self.rect.topLeft()
        return QtCore.QPoint(pos.x() / px, pos.y() / py)

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


class ScaledImageSliceItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        image: ScaledImageItem,
        pen: QtGui.QPen = None,
        font: QtGui.QFont = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(parent)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.white, 2.0, QtCore.Qt.DotLine)
            pen.setCosmetic(True)
            pen.setCapStyle(QtCore.Qt.RoundCap)

        if font is None:
            font = QtGui.QFont()

        self.image = image

        self.pen = pen
        self.font = font

        self.line = QtCore.QLineF()

        self.data: np.ndarray = None
        self.poly = QtGui.QPolygonF()

        self.action_copy_to_clipboard = qAction(
            "copy-text",
            "Copy To Clipboard",
            "Copy slice values to the clipboard.",
            self.actionCopyToClipboard,
        )

    def actionCopyToClipboard(self):
        if self.data is None:
            return
        html = (
            '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
            "<table>"
        )
        text = ""
        for x in self.data:
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
        if points.size > 2:
            self.data = self.image.image._array[points[:, 1], points[:, 0]]

            xs = np.linspace(0.0, self.line.length(), self.data.size)
            ys = -1.0 * normalise(self.data, 0.0, height)

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
        view = next(iter(self.scene().views()), None)
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
        if self.line.dx() < 0.0:
            p1, p2 = p2, p1
        painter.drawLine(p1, self.poly.first())
        painter.drawLine(p2, self.poly.last())

        pen = QtGui.QPen(self.pen)
        pen.setStyle(QtCore.Qt.SolidLine)
        painter.setPen(pen)
        painter.drawPolyline(self.poly)

        if not self.line.p1().isNull():
            pen.setWidth(10)
            painter.setPen(pen)
            painter.drawPoints([self.line.p1(), self.line.p2()])
            painter.setPen(self.pen)

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()
        menu.addAction(self.action_copy_to_clipboard)
        menu.exec_(event.screenPos())

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.line.setPoints(event.pos(), event.pos())
            self.data = None
            self.poly = QtGui.QPolygonF()
            self.prepareGeometryChange()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.line.setP2(event.pos())
            self.createSlicePoly()
            self.prepareGeometryChange()
        super().mouseMoveEvent(event)
