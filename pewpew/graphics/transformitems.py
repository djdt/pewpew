from PySide6 import QtCore, QtGui, QtWidgets

from typing import Optional, Tuple


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
        super().__init__(item)

        self.transform_handle: Optional[Tuple[str, str]] = None
        self.handle_size = handle_size

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
            dx = event.scenePos().x() - event.lastScenePos().x()
            dy = event.scenePos().y() - event.lastScenePos().y()
            self.parentItem().moveBy(dx, dy)
            return

        poly = self.parentItem().mapToScene(self.corners())
        if self.transformHandle[0] == "corner":
            square = QtGui.QTransform.quadToSquare(poly)
            new_pos = square.map(event.scenePos())  # type: ignore
            unit_rect = QtCore.QRectF(0.0, 0.0, 1.0, 1.0)
            if self.transformHandle[1] == "topLeft":
                unit_rect.setTopLeft(new_pos)
            elif self.transformHandle[1] == "topRight":
                unit_rect.setTopRight(new_pos)
            elif self.transformHandle[1] == "bottomRight":
                unit_rect.setBottomRight(new_pos)
            elif self.transformHandle[1] == "bottomLeft":
                unit_rect.setBottomLeft(new_pos)

            if not QtCore.Qt.ShiftModifier & event.modifiers():
                if "top" in self.transformHandle[1]:
                    unit_rect.setWidth(unit_rect.height())
                else:
                    unit_rect.setHeight(unit_rect.width())
                if self.transformHandle[1] == "topLeft":
                    unit_rect.moveBottomRight(QtCore.QPointF(1.0, 1.0))

            invert, ok = square.inverted()  # type: ignore
            if not ok:
                raise ValueError("Invalid inverse square transformation.")

            poly = invert.map(rectf_to_polygonf(unit_rect))

        elif self.transformHandle[0] == "edge":
            center = self.parentItem().mapToScene(self.center())
            edge = self.parentItem().mapToScene(
                self.edges().at(self.edge_order.index(self.transformHandle[1]))
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
        self.transformHandle = None

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
