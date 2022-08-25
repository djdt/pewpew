from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from typing import List, Optional, Tuple


def rectf_to_polygonf(rect: QtCore.QRectF) -> QtGui.QPolygonF:
    return QtGui.QPolygonF(
        [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
    )


class TransformHandles(QtWidgets.QGraphicsObject):
    corner_order = ["topLeft", "topRight", "bottomRight", "bottomLeft"]
    edge_order = ["top", "right", "bottom", "left"]

    transform_cursors = {
        "translate": QtCore.Qt.ClosedHandCursor,
        "scale": QtCore.Qt.SizeAllCursor,
        "rotate": QtCore.Qt.PointingHandCursor,
    }

    def __init__(
        self,
        item: QtWidgets.QGraphicsItem,
        anchors: List[str] = ["translate", "scale", "rotate"],
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.item = item
        # self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
        self.setAcceptedMouseButtons(QtCore.Qt.LeftButton | QtCore.Qt.MiddleButton)
        self.setZValue(self.item.zValue() + 10.0)

        self.use_anchors = anchors

        self.transform_mode: str = "none"
        self.transform_anchor: str = "none"

        self.handle_size = 12

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
        return self.item.transform().mapRect(self.item.boundingRect()).center()

    def boundingRect(self) -> QtCore.QRectF:
        rect = self.item.boundingRect()
        view = next(iter(self.scene().views()))
        if view is not None:
            adjust = view.mapToScene(
                QtCore.QRect(0, 0, self.handle_size, self.handle_size)
            ).boundingRect()
            rect = rect.adjusted(
                -adjust.width(), -adjust.height(), adjust.width(), adjust.height()
            )
        return self.item.transform().mapRect(rect).normalized()

    def corners(self) -> QtGui.QPolygonF:
        # QPolygonF(rect) return 5 points
        poly = rectf_to_polygonf(self.item.boundingRect())
        return self.item.transform().map(poly)

    def edges(self) -> QtGui.QPolygonF:
        rect = self.item.boundingRect()
        center = rect.center()
        poly = QtGui.QPolygonF(
            [
                QtCore.QPointF(center.x(), rect.top()),
                QtCore.QPointF(rect.right(), center.y()),
                QtCore.QPointF(center.x(), rect.bottom()),
                QtCore.QPointF(rect.left(), center.y()),
            ]
        )
        return self.item.transform().map(poly)

    def closest_point(self, pos: QtCore.QPointF) -> Tuple[str, str, float]:
        result = ("center", "center", QtCore.QLineF(pos, self.center()).length())

        for corner, name in zip(self.corners(), self.corner_order):
            dist = QtCore.QLineF(pos, corner).length()
            if dist < result[2]:
                result = ("corner", name, dist)

        for edge, name in zip(self.edges(), self.edge_order):
            dist = QtCore.QLineF(pos, edge).length()
            if dist < result[2]:
                result = ("edge", name, dist)

        return result

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.transform_mode = "none"
        self.transform_anchor = "none"

        if event.button() & QtCore.Qt.LeftButton:
            closest = self.closest_point(event.scenePos())

            view = next(iter(self.scene().views()))
            if view is not None:
                max_dist = (
                    view.mapToScene(
                        QtCore.QRect(0, 0, self.handle_size, self.handle_size)
                    )
                    .boundingRect()
                    .width()
                )
            else:
                event.ignore()
                return

            if closest[2] < max_dist:
                if closest[0] == "center" and "translate" in self.use_anchors:  # Dont use translate
                    self.transform_mode = "translate"
                    self.transform_anchor = "center"
                elif closest[0] == "corner" and "scale" in self.use_anchors:
                    self.transform_mode = "scale"
                    self.transform_anchor = closest[1]
                elif closest[0] == "edge" and "rotate" in self.use_anchors:
                    self.transform_mode = "rotate"
                    self.transform_anchor = closest[1]
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if not event.buttons() & (QtCore.Qt.LeftButton | QtCore.Qt.MiddleButton):
            return
        if self.transform_mode == "none":
            return

        dx = event.scenePos().x() - event.lastScenePos().x()
        dy = event.scenePos().y() - event.lastScenePos().y()

        self.setCursor(self.transform_cursors[self.transform_mode])

        poly = self.corners()
        transform = self.item.transform()
        if self.transform_mode == "translate":
            poly.translate(dx, dy)
        elif self.transform_mode == "scale":
            square = QtGui.QTransform.quadToSquare(poly)
            if square is None:
                logger.warning("Invalid quadToSquare transformation.")
                return
            new_pos = square.map(event.scenePos())
            unit_rect = QtCore.QRectF(0.0, 0.0, 1.0, 1.0)

            if self.transform_anchor == "topLeft":
                unit_rect.setTopLeft(new_pos)
            elif self.transform_anchor == "topRight":
                unit_rect.setTopRight(new_pos)
            elif self.transform_anchor == "bottomRight":
                unit_rect.setBottomRight(new_pos)
            elif self.transform_anchor == "bottomLeft":
                unit_rect.setBottomLeft(new_pos)
            invert, ok = square.inverted()
            if not ok:
                logger.warning("Invalid inverse square transformation.")
                return
            poly = invert.map(rectf_to_polygonf(unit_rect))
        elif self.transform_mode == "rotate":
            center = self.center()
            edge = self.edges()[self.edge_order.index(self.transform_anchor)]
            angle = QtCore.QLineF(center, edge).angleTo(
                QtCore.QLineF(center, event.scenePos())
            )
            if QtCore.Qt.ShiftModifier == event.modifiers():
                angle = np.round(angle / 45.0) * 45.0

            lines = [QtCore.QLineF(center, p) for p in poly]
            for line in lines:
                line.setAngle(line.angle() + angle)
            poly = QtGui.QPolygonF([line.p2() for line in lines])

        rect = self.item.boundingRect()
        transform = QtGui.QTransform.quadToQuad(rectf_to_polygonf(rect), poly)
        if transform is None:
            logger.warning("Invalid quadToQuad transformation.")
            return
        # Discard non affine transforms
        if not transform.isAffine():
            logger.info("Not affine transform, discarding m31, m32, m33.")
            transform = QtGui.QTransform(
                transform.m11(),
                transform.m12(),
                transform.m21(),
                transform.m22(),
                transform.dx(),
                transform.dy(),
            )
        self.item.setTransform(transform)

        self.prepareGeometryChange()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() not in [QtCore.Qt.LeftButton, QtCore.Qt.MiddleButton]:
            return
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.transform_mode = "none"

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
        if "translate" in self.use_anchors:
            painter.drawPoint(center)
        if "scale" in self.use_anchors:
            painter.drawPoints(corners)

        if "rotate" in self.use_anchors:
            pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(edges)

        pen.setCapStyle(QtCore.Qt.SquareCap)
        pen.setColor(QtGui.QColor(255, 255, 255))
        pen.setWidth(self.handle_size)
        painter.setPen(pen)
        if "transform" in self.use_anchors:
            painter.drawPoint(center)
        if "scale" in self.use_anchors:
            painter.drawPoints(corners)

        if "rotate" in self.use_anchors:
            pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(edges)

        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawConvexPolygon(corners)

        painter.restore()
