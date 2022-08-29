from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from typing import List, Optional, Tuple


def rectf_to_polygonf(rect: QtCore.QRectF) -> QtGui.QPolygonF:
    # QtGui.QPolygonF(rect) returns 5 points, not per qt documentation
    # topLeft, topRight, bottomRight, bottomLeft
    poly = QtGui.QPolygonF(rect)
    poly.removeLast()
    return poly


class TransformHandlesItem(QtWidgets.QGraphicsObject):
    """Creates the selected transform handles over an items bounding box.
    These will affect the item.transform()"""

    corner_order = ["topLeft", "topRight", "bottomRight", "bottomLeft"]
    edge_order = ["top", "right", "bottom", "left"]

    cursors = {
        "left": QtCore.Qt.SizeHorCursor,
        "right": QtCore.Qt.SizeHorCursor,
        "top": QtCore.Qt.SizeVerCursor,
        "bottom": QtCore.Qt.SizeVerCursor,
        "topLeft": QtCore.Qt.SizeFDiagCursor,
        "topRight": QtCore.Qt.SizeBDiagCursor,
        "bottomLeft": QtCore.Qt.SizeBDiagCursor,
        "bottomRight": QtCore.Qt.SizeFDiagCursor,
    }

    def __init__(
        self,
        item: QtWidgets.QGraphicsItem,
        handle_size: int = 12,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        # self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)

        self.transform_handle: Optional[Tuple[str, str]] = None
        self.handle_size = handle_size
        super().__init__(item)

        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.MiddleButton)
        self.setZValue(item.zValue() + 1.0)

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

    def center(self) -> QtCore.QPointF:
        return self.parentItem().boundingRect().center()

    def boundingRect(self) -> QtCore.QRectF:
        rect = self.parentItem().boundingRect()
        adjust = self.maxHandleDist()
        # rect = self.item.boundingRect()
        # view = next(iter(self.scene().views()))
        # if view is not None:
        #     adjust = view.mapToScene(
        #         QtCore.QRect(0, 0, self.handle_size, self.handle_size)
        #     ).boundingRect()
        rect = rect.adjusted(-adjust, -adjust, adjust, adjust)
        return rect

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
            .mapRect(QtCore.QRect(0, 0, self.handle_size, self.handle_size))
            .width()
        )

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

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        handle = self.handleAt(event.pos())
        if handle is None:
            self.unsetCursor()
        else:
            self.setCursor(self.cursors[handle[1]])
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        # self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() & QtCore.Qt.LeftButton:
            self.transformHandle = self.handleAt(event.pos())
            if self.transformHandle is not None:
                event.accept()
                self.setCursor(self.cursors[self.transformHandle[1]])
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.buttons() & (QtCore.Qt.LeftButton | QtCore.Qt.MiddleButton):
            return
        if self.transformHandle is None:
            return

        dx = event.scenePos().x() - event.lastScenePos().x()
        dy = event.scenePos().y() - event.lastScenePos().y()

        rect = self.parentItem().boundingRect()
        poly = self.parentItem().mapToScene(self.corners())

        square = QtGui.QTransform.quadToSquare(poly)
        new_pos = square.map(event.scenePos())
        unit_rect = QtCore.QRectF(0.0, 0.0, 1.0, 1.0)

        if self.transformHandle[1] == "topLeft":
            unit_rect.setTopLeft(new_pos)
        elif self.transformHandle[1] == "topRight":
            unit_rect.setTopRight(new_pos)
        elif self.transformHandle[1] == "bottomRight":
            unit_rect.setBottomRight(new_pos)
        elif self.transformHandle[1] == "bottomLeft":
            unit_rect.setBottomLeft(new_pos)
        invert, ok = square.inverted()
        if not ok:
            logger.warning("Invalid inverse square transformation.")
            return
        poly = invert.map(rectf_to_polygonf(unit_rect))
        transform = QtGui.QTransform.quadToQuad(self.corners(), poly)
        self.parentItem().setTransform(transform)
        # self.parentItem().prepareGeometryChange()
        # # transform = self.item.transform()
        # if self.transform_mode == "translate":
        #     poly.translate(dx, dy)
        # elif self.transform_mode == "scale":
        #     square = QtGui.QTransform.quadToSquare(poly)
        #     if square is None:
        #         logger.warning("Invalid quadToSquare transformation.")
        #         return
        #     new_pos = square.map(event.pos())
        #     unit_rect = QtCore.QRectF(0.0, 0.0, 1.0, 1.0)

        #     if self.transform_anchor == "topLeft":
        #         unit_rect.setTopLeft(new_pos)
        #     elif self.transform_anchor == "topRight":
        #         unit_rect.setTopRight(new_pos)
        #     elif self.transform_anchor == "bottomRight":
        #         unit_rect.setBottomRight(new_pos)
        #     elif self.transform_anchor == "bottomLeft":
        #         unit_rect.setBottomLeft(new_pos)
        #     invert, ok = square.inverted()
        #     if not ok:
        #         logger.warning("Invalid inverse square transformation.")
        #         return
        #     poly = invert.map(rectf_to_polygonf(unit_rect))
        # elif self.transform_mode == "rotate":
        #     center = self.center()
        #     edge = self.edges()[self.edge_order.index(self.transform_anchor)]
        #     angle = QtCore.QLineF(center, edge).angleTo(
        #         QtCore.QLineF(center, event.pos())
        #     )
        #     if QtCore.Qt.ShiftModifier == event.modifiers():
        #         angle = np.round(angle / 45.0) * 45.0

        #     lines = [QtCore.QLineF(center, p) for p in poly]
        #     for line in lines:
        #         line.setAngle(line.angle() + angle)
        #     poly = QtGui.QPolygonF([line.p2() for line in lines])

        # rect = self.parentItem().boundingRect()
        # transform = QtGui.QTransform.quadToQuad(rectf_to_polygonf(rect), poly)
        # if transform is None:
        #     logger.warning("Invalid quadToQuad transformation.")
        #     return
        # # Discard non affine transforms
        # if not transform.isAffine():
        #     logger.info("Not affine transform, discarding m31, m32, m33.")
        #     transform = QtGui.QTransform(
        #         transform.m11(),
        #         transform.m12(),
        #         transform.m21(),
        #         transform.m22(),
        #         transform.dx(),
        #         transform.dy(),
        #     )
        # self.parentItem().setTransform(transform)

        # self.prepareGeometryChange()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() not in [QtCore.Qt.LeftButton, QtCore.Qt.MiddleButton]:
            return
        self.unsetCursor()
        self.transformHandle = None

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        center = self.center()
        corners = self.corners()
        edges = self.edges()

        pen = QtGui.QPen(QtGui.QColor(0, 0, 0), self.handle_size + 2)
        pen.setCosmetic(True)

        painter.setPen(pen)
        # if "translate" in self.use_anchors:
        #     painter.drawPoint(center)
        # if "scale" in self.use_anchors:
        painter.drawPoints(corners)

        # if "rotate" in self.use_anchors:
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPoints(edges)

        pen.setCapStyle(QtCore.Qt.SquareCap)
        pen.setColor(QtGui.QColor(255, 255, 255))
        pen.setWidth(self.handle_size)
        painter.setPen(pen)
        # if "transform" in self.use_anchors:
        # painter.drawPoint(center)
        # if "scale" in self.use_anchors:
        painter.drawPoints(corners)

        # if "rotate" in self.use_anchors:
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPoints(edges)

        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawConvexPolygon(corners)

        painter.restore()
