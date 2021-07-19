from PySide2 import QtCore, QtGui, QtWidgets

from typing import Optional


class ResizeableRectItem(QtWidgets.QGraphicsRectItem):
    cursors = {
        "left": QtCore.Qt.SizeHorCursor,
        "right": QtCore.Qt.SizeHorCursor,
        "top": QtCore.Qt.SizeVerCursor,
        "bottom": QtCore.Qt.SizeVerCursor,
        "topleft": QtCore.Qt.SizeFDiagCursor,
        "topright": QtCore.Qt.SizeBDiagCursor,
        "bottomleft": QtCore.Qt.SizeBDiagCursor,
        "bottomright": QtCore.Qt.SizeFDiagCursor,
    }

    def __init__(
        self,
        rect: QtCore.QRectF,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(rect, parent)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self.selected_edge: Optional[str] = None

    def boundingRect(self) -> QtCore.QRectF:
        rect = super().boundingRect()
        view = next(iter(self.scene().views()), None)
        if view is None:
            return rect

        dist = view.mapToScene(QtCore.QRect(0, 0, 10, 1)).boundingRect().width()
        return rect.marginsAdded(QtCore.QMarginsF(dist, dist, dist, dist))

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        rect = self.boundingRect()
        path.addRect(rect)
        return path

    def edgeAt(self, pos: QtCore.QPointF) -> Optional[str]:
        view = next(iter(self.scene().views()))
        dist = view.mapToScene(QtCore.QRect(0, 0, 10, 1)).boundingRect().width()

        edge = ""
        if abs(self.rect().top() - pos.y()) < dist:
            edge += "top"
        elif abs(self.rect().bottom() - pos.y()) < dist:
            edge += "bottom"
        if abs(self.rect().left() - pos.x()) < dist:
            edge += "left"
        elif abs(self.rect().right() - pos.x()) < dist:
            edge += "right"
        if edge == "":
            return None
        return edge

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if self.isSelected():
            edge = self.edgeAt(event.pos())
            if edge is None:
                self.setCursor(QtCore.Qt.ArrowCursor)
            else:
                self.setCursor(self.cursors[edge])
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if self.isSelected():
            self.setCursor(QtCore.Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if self.isSelected():
            self.selected_edge = self.edgeAt(event.pos())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        self.selected_edge = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        pos = self.itemChange(QtWidgets.QGraphicsItem.ItemPositionChange, event.pos())
        if self.selected_edge is None:
            super().mouseMoveEvent(event)
        else:
            rect = self.rect()
            if self.selected_edge.startswith("top") and pos.y() < rect.bottom():
                rect.setTop(pos.y())
            elif self.selected_edge.startswith("bottom") and pos.y() > rect.top():
                rect.setBottom(pos.y())
            if self.selected_edge.endswith("left") and pos.x() < rect.right():
                rect.setLeft(pos.x())
            elif self.selected_edge.endswith("right") and pos.x() > rect.left():
                rect.setRight(pos.x())

            self.prepareGeometryChange()
            self.setRect(rect)
