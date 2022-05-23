from PySide2 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction
from pewpew.graphics.aligneditems import UnscaledAlignedTextItem

from typing import Optional


class EditableLabelItem(UnscaledAlignedTextItem):
    labelChanged = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtWidgets.QGraphicsItem,
        text: str,
        label_text: str,
        alignment: Optional[QtCore.Qt.Alignment] = None,
        font: Optional[QtGui.QFont] = None,
        brush: Optional[QtGui.QBrush] = None,
        pen: Optional[QtGui.QPen] = None,
    ):
        super().__init__(parent, text, alignment, font, brush, pen)
        self.label = label_text

    def editLabel(self) -> QtWidgets.QInputDialog:
        """Simple dialog for editing the label."""
        dlg = QtWidgets.QInputDialog(self.scene().views()[0])
        dlg.setWindowTitle(f"{self.label} Label")
        dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
        dlg.setTextValue(self.text())
        dlg.setLabelText(f"{self.label}:")
        dlg.textValueSelected.connect(self.labelChanged)
        dlg.open()

        return dlg

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        action_edit = qAction(
            "edit-rename",
            f"Set {self.label}",
            "Set the text of the label.",
            self.editLabel,
        )
        action_hide = qAction(
            "visibility",
            f"Hide {self.label} Label",
            "Hide the laser name label.",
            self.hide,
        )
        menu = QtWidgets.QMenu()
        menu.addAction(action_edit)
        menu.addAction(action_hide)
        menu.exec_(event.screenPos())
        event.accept()


class ResizeableRectItem(QtWidgets.QGraphicsObject):
    """A mouse resizable rectangle.

    Click and drag the corners or edges of the rectangle to resize.
    """

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
        cursor_dist: int = 6,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self.rect = rect
        self.pen = QtGui.QPen()
        self.brush = QtGui.QBrush()

        self.selected_edge: Optional[str] = None
        self.cursor_dist = cursor_dist

    def setBrush(self, brush: QtGui.QBrush) -> None:
        self.brush = brush

    def setPen(self, pen: QtGui.QPen) -> None:
        self.pen = pen

    def boundingRect(self) -> QtCore.QRectF:
        view = next(iter(self.scene().views()), None)
        if view is None:
            return self.rect

        dist = (
            view.mapToScene(QtCore.QRect(0, 0, self.cursor_dist, 1))
            .boundingRect()
            .width()
        )
        return self.rect.marginsAdded(QtCore.QMarginsF(dist, dist, dist, dist))

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        painter.save()
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        painter.drawRect(self.rect)
        painter.restore()

    def edgeAt(self, pos: QtCore.QPointF) -> Optional[str]:
        view = next(iter(self.scene().views()))
        dist = (
            view.mapToScene(QtCore.QRect(0, 0, self.cursor_dist, 1))
            .boundingRect()
            .width()
        )

        edge = ""
        if abs(self.rect.top() - pos.y()) < dist:
            edge += "top"
        elif abs(self.rect.bottom() - pos.y()) < dist:
            edge += "bottom"
        if abs(self.rect.left() - pos.x()) < dist:
            edge += "left"
        elif abs(self.rect.right() - pos.x()) < dist:
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
        if self.selected_edge is None:
            super().mouseMoveEvent(event)
        else:
            pos = event.pos()
            if self.selected_edge.startswith("top") and pos.y() < self.rect.bottom():
                self.rect.setTop(pos.y())
            elif self.selected_edge.startswith("bottom") and pos.y() > self.rect.top():
                self.rect.setBottom(pos.y())
            if self.selected_edge.endswith("left") and pos.x() < self.rect.right():
                self.rect.setLeft(pos.x())
            elif self.selected_edge.endswith("right") and pos.x() > self.rect.left():
                self.rect.setRight(pos.x())

            self.prepareGeometryChange()
