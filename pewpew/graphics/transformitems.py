from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np

from typing import List, Optional, Tuple

# Todo affine transform not work with position change


def rectf_to_polygonf(rect: QtCore.QRectF) -> QtGui.QPolygonF:
    # QtGui.QPolygonF(rect) returns 5 points, not per qt documentation
    # topLeft, topRight, bottomRight, bottomLeft
    poly = QtGui.QPolygonF(rect)
    poly.removeLast()
    return poly


class TransformItem(QtWidgets.QGraphicsObject):
    def __init__(self, item: QtWidgets.QGraphicsItem, handle_size: int):
        super().__init__(item)

        self.handle_size = handle_size
        self.initial_transform = QtGui.QTransform(self.parentItem().transform())

        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.MiddleButton)
        self.setZValue(item.zValue() + 1.0)

    def boundingRect(self) -> QtCore.QRectF:
        rect = self.parentItem().boundingRect()
        return rect

    def center(self) -> QtCore.QPointF:
        return self.parentItem().boundingRect().center()

    def corners(self) -> QtGui.QPolygonF:
        return rectf_to_polygonf(self.parentItem().boundingRect())

    def edges(self) -> QtGui.QPolygonF:
        rect = self.parentItem().boundingRect()
        center = rect.center()
        poly = QtGui.QPolygonF(
            [
                QtCore.QPointF(center.x(), rect.top()),
                QtCore.QPointF(rect.right(), center.y()),
                QtCore.QPointF(center.x(), rect.bottom()),
                QtCore.QPointF(rect.left(), center.y()),
            ]
        )
        return poly

    def maxHandleDist(self) -> float:
        view = next(iter(self.scene().views()))
        if view is None:
            return self.handle_size
        return (
            self.parentItem()
            .deviceTransform(view.transform())
            .inverted()[0]
            .mapRect(QtCore.QRect(0, 0, self.handle_size, self.handle_size))  # type: ignore
            .width()
        )


class AffineTransformItem(TransformItem):
    def __init__(self, item: QtWidgets.QGraphicsItem, handle_size: int = 12):
        super().__init__(item, handle_size)

        # List of (start, end) points for affine transform
        self.handles: List[QtCore.QPointF] = []
        self.transform_handle: Optional[int] = None

    def boundingRect(self) -> QtCore.QRectF:
        adjust = self.maxHandleDist()
        poly = self.parentItem().mapFromScene(self.handles)
        return (
            super()
            .boundingRect()
            .united(poly.boundingRect())
            .adjusted(-adjust, -adjust, adjust, adjust)
        )

    def shape(self) -> QtGui.QPainterPath:
        corners = self.corners()
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.WindingFill)
        path.addPolygon(corners)

        view = next(iter(self.scene().views()))
        if view is not None:
            adjust = (
                view.mapToScene(QtCore.QRect(0, 0, self.handle_size, self.handle_size))
                .boundingRect()
                .width()
            )

            poly = self.parentItem().mapFromScene(self.handles)
            for point in poly:  # type: ignore
                path.addEllipse(point, adjust, adjust)

        return path

    def handleAt(self, pos: QtCore.QPointF) -> Optional[int]:
        max_dist = self.maxHandleDist()
        result = None

        for i, point in enumerate(self.handles):
            dist = QtCore.QLineF(pos, point).length()
            if dist - 1e-5 < max_dist:
                result = i
                max_dist = dist

        return result

    def calculateTransform(self) -> QtGui.QTransform:
        if len(self.handles) != 6:
            return QtGui.QTransform()

        # Remove position offset
        pos = self.parentItem().pos()

        xs = [p.x() - pos.x() for p in self.handles[::2]]
        ys = [p.y() - pos.y() for p in self.handles[::2]]
        us = [p.x() - pos.x() for p in self.handles[1::2]]
        vs = [p.y() - pos.y() for p in self.handles[1::2]]

        A = np.array([xs, ys, np.ones(3)])
        B = np.array([us, vs, np.ones(3)])
        C = np.dot(B, np.linalg.inv(A))

        return QtGui.QTransform(C[0, 0], C[1, 0], C[0, 1], C[1, 1], C[0, 2], C[1, 2])

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() & QtCore.Qt.LeftButton:
            if len(self.handles) < 6:
                self.handles.extend([event.scenePos(), event.scenePos()])
                self.prepareGeometryChange()
                event.accept()
                return

            self.transform_handle = self.handleAt(event.scenePos())
            if self.transform_handle is not None:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.buttons() & (QtCore.Qt.LeftButton | QtCore.Qt.MiddleButton):
            return

        if self.transform_handle is None:
            return

        self.handles[self.transform_handle] = event.scenePos()

        if self.transform_handle % 2 == 1:
            self.parentItem().setTransform(self.calculateTransform())
        else:
            self.parentItem().setTransform(self.initial_transform)

        self.prepareGeometryChange()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() not in [QtCore.Qt.LeftButton, QtCore.Qt.MiddleButton]:
            return
        if self.transform_handle is not None and self.transform_handle % 2 == 0:
            self.parentItem().setTransform(self.calculateTransform())
            self.prepareGeometryChange()

        self.unsetCursor()
        self.transform_handle = None

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        starts = self.parentItem().mapFromScene(self.handles[::2])
        ends = self.parentItem().mapFromScene(self.handles[1::2])

        pen = QtGui.QPen(QtGui.QColor(0, 0, 0), 2.0, QtCore.Qt.DotLine)
        pen.setCosmetic(True)

        painter.setPen(pen)
        for s, e in zip(starts, ends):  # type: ignore
            painter.drawLine(s, e)

        pen.setWidth(self.handle_size + 2)

        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPoints(starts)

        pen.setCapStyle(QtCore.Qt.SquareCap)
        painter.setPen(pen)
        painter.drawPoints(ends)

        pen.setWidth(self.handle_size)

        pen.setColor(QtGui.QColor(128, 128, 128))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPoints(starts)

        pen.setColor(QtGui.QColor(255, 255, 255))
        pen.setCapStyle(QtCore.Qt.SquareCap)
        painter.setPen(pen)
        painter.drawPoints(ends)

        pen.setWidth(1)
        pen.setStyle(QtCore.Qt.SolidLine)
        painter.setPen(pen)
        painter.drawConvexPolygon(rectf_to_polygonf(self.parentItem().boundingRect()))

        painter.restore()


class ScaleRotateTransformItem(TransformItem):
    """Creates the selected transform handles over an items bounding box.
    These will affect the item.transform()"""

    corner_order = ["topLeft", "topRight", "bottomRight", "bottomLeft"]
    edge_order = ["top", "right", "bottom", "left"]

    cursors = {
        "left": QtCore.Qt.ClosedHandCursor,
        "right": QtCore.Qt.ClosedHandCursor,
        "top": QtCore.Qt.ClosedHandCursor,
        "bottom": QtCore.Qt.ClosedHandCursor,
        "topLeft": QtCore.Qt.SizeFDiagCursor,
        "topRight": QtCore.Qt.SizeBDiagCursor,
        "bottomLeft": QtCore.Qt.SizeBDiagCursor,
        "bottomRight": QtCore.Qt.SizeFDiagCursor,
    }

    def __init__(self, item: QtWidgets.QGraphicsItem, handle_size: int = 12):
        super().__init__(item, handle_size)

        self.transform_handle: Optional[Tuple[str, str]] = None

    def shape(self) -> QtGui.QPainterPath:
        corners = self.corners()
        edges = self.edges()
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.WindingFill)
        path.addPolygon(corners)

        view = next(iter(self.scene().views()))
        if view is not None:
            adjust = (
                view.mapToScene(QtCore.QRect(0, 0, self.handle_size, self.handle_size))
                .boundingRect()
                .width()
            )

            for point in corners + edges:
                path.addEllipse(point, adjust, adjust)

        return path

    def boundingRect(self) -> QtCore.QRectF:
        rect = self.parentItem().boundingRect()
        adjust = self.maxHandleDist()
        rect = rect.adjusted(-adjust, -adjust, adjust, adjust)
        return rect

    def handleAt(self, pos: QtCore.QPointF) -> Optional[Tuple[str, str]]:
        max_dist = self.maxHandleDist()

        result = None

        for corner, name in zip(self.corners(), self.corner_order):
            dist = QtCore.QLineF(pos, corner).length()
            if dist < max_dist:
                result = ("corner", name)
                max_dist = dist

        for edge, name in zip(self.edges(), self.edge_order):
            dist = QtCore.QLineF(pos, edge).length()
            if dist < max_dist:
                result = ("edge", name)
                max_dist = dist

        return result

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() & QtCore.Qt.LeftButton:
            self.transform_handle = self.handleAt(event.pos())
            if self.transform_handle is not None:
                event.accept()
                self.setCursor(self.cursors[self.transform_handle[1]])
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.buttons() & (QtCore.Qt.LeftButton | QtCore.Qt.MiddleButton):
            return

        if self.transform_handle is None:
            dx = event.scenePos().x() - event.lastScenePos().x()
            dy = event.scenePos().y() - event.lastScenePos().y()
            self.parentItem().moveBy(dx, dy)
            return

        poly = self.parentItem().mapToScene(self.corners())
        if self.transform_handle[0] == "corner":
            square = QtGui.QTransform.quadToSquare(poly)
            new_pos = square.map(event.scenePos())  # type: ignore
            unit_rect = QtCore.QRectF(0.0, 0.0, 1.0, 1.0)
            if self.transform_handle[1] == "topLeft":
                unit_rect.setTopLeft(new_pos)
            elif self.transform_handle[1] == "topRight":
                unit_rect.setTopRight(new_pos)
            elif self.transform_handle[1] == "bottomRight":
                unit_rect.setBottomRight(new_pos)
            elif self.transform_handle[1] == "bottomLeft":
                unit_rect.setBottomLeft(new_pos)

            if not QtCore.Qt.ShiftModifier & event.modifiers():
                if "top" in self.transform_handle[1]:
                    unit_rect.setWidth(unit_rect.height())
                else:
                    unit_rect.setHeight(unit_rect.width())
                if self.transform_handle[1] == "topLeft":
                    unit_rect.moveBottomRight(QtCore.QPointF(1.0, 1.0))

            invert, ok = square.inverted()  # type: ignore
            if not ok:
                raise ValueError("Invalid inverse square transformation.")

            poly = invert.map(rectf_to_polygonf(unit_rect))

        elif self.transform_handle[0] == "edge":
            center = self.parentItem().mapToScene(self.center())
            edge = self.parentItem().mapToScene(
                self.edges().at(self.edge_order.index(self.transform_handle[1]))
            )

            line_to = QtCore.QLineF(center, event.scenePos())
            if QtCore.Qt.ShiftModifier & event.modifiers():
                line_to.setAngle(line_to.angle() - line_to.angle() % 45)
            angle = QtCore.QLineF(center, edge).angleTo(line_to)

            lines = [QtCore.QLineF(center, p) for p in poly]
            for line in lines:
                line.setAngle(line.angle() + angle)
            poly = QtGui.QPolygonF([line.p2() for line in lines])

        transform = QtGui.QTransform.quadToQuad(
            self.corners(), poly.translated(-self.parentItem().pos())
        )
        self.parentItem().setTransform(transform)
        self.prepareGeometryChange()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() not in [QtCore.Qt.LeftButton, QtCore.Qt.MiddleButton]:
            return
        self.unsetCursor()
        self.transform_handle = None

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # center = self.center()
        corners = self.corners()
        edges = self.edges()

        pen = QtGui.QPen(QtGui.QColor(0, 0, 0), self.handle_size + 2)
        pen.setCosmetic(True)

        # Outside
        painter.setPen(pen)
        painter.drawPoints(corners)

        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPoints(edges)

        # Inside
        pen.setCapStyle(QtCore.Qt.SquareCap)
        pen.setColor(QtGui.QColor(255, 255, 255))
        pen.setWidth(self.handle_size)
        painter.setPen(pen)
        painter.drawPoints(corners)

        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPoints(edges)

        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawConvexPolygon(corners)

        painter.restore()
