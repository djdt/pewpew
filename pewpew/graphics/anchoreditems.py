from PySide2 import QtCore, QtGui, QtWidgets

from typing import Optional, Union


class AnchoredItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        anchor: Union[QtCore.Qt.AnchorPoint, QtCore.Qt.Corner],
        alignment: QtCore.Qt.Alignment,
        parent: QtWidgets.QGraphicsItem,
    ):
        super().__init__(parent=parent)

        if anchor is None:
            anchor = QtCore.Qt.TopLeftCorner
        if alignment is None:
            alignment = QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft

        self.anchor = anchor
        self.alignment = alignment

    def anchorPos(
        self, anchor: Union[QtCore.Qt.AnchorPoint, QtCore.Qt.Corner]
    ) -> QtCore.QPointF:
        rect = self.parentItem().boundingRect()

        if self.flags() & QtWidgets.QGraphicsItem.ItemIgnoresTransformations:
            view = next(iter(self.scene().views()))
            rect = QtCore.QRectF(view.mapFromScene(rect).boundingRect())
            rect.moveTo(self.parentItem().boundingRect().topLeft())

        if isinstance(anchor, QtCore.Qt.Corner):
            if anchor == QtCore.Qt.TopLeftCorner:
                pos = rect.topLeft()
            elif anchor == QtCore.Qt.TopRightCorner:
                pos = rect.topRight()
            elif anchor == QtCore.Qt.BottomLeftCorner:
                pos = rect.bottomLeft()
            else:  # BottomRightCorner
                pos = rect.bottomRight()
        else:  # AnchorPoint
            if anchor == QtCore.Qt.AnchorTop:
                pos = QtCore.QPointF(rect.center().x(), rect.top())
            elif anchor == QtCore.Qt.AnchorLeft:
                pos = QtCore.QPointF(rect.left(), rect.center().y())
            elif anchor == QtCore.Qt.AnchorRight:
                pos = QtCore.QPointF(rect.right(), rect.center().y())
            elif anchor == QtCore.Qt.AnchorBottom:
                pos = QtCore.QPointF(rect.center().x(), rect.bottom())
            else:
                raise ValueError("Only Top, Left, Right, Bottom anchors supported.")
        return pos

    def alignedPos(
        self, pos: QtCore.QPointF, alignment: QtCore.Qt.Alignment
    ) -> QtCore.QPointF:
        rect = self.boundingRect()

        if alignment & QtCore.Qt.AlignHCenter:
            pos.setX(pos.x() - rect.width() / 2.0)
        elif alignment & QtCore.Qt.AlignRight:
            pos.setX(pos.x() - rect.width())

        if alignment & QtCore.Qt.AlignVCenter:
            pos.setY(pos.y() - rect.height() / 2.0)
        elif alignment & QtCore.Qt.AlignBottom:
            pos.setY(pos.y() - rect.height())

        return pos
    
    def unalignedRect(self) -> QtCore.QRectF:
        raise NotImplementedError

    def boundingRect(self) -> QtCore.QRectF:
        rect = self.unalignedRect()

        if self.flags() & QtWidgets.QGraphicsItem.ItemIgnoresTransformations:
            view = next(iter(self.scene().views()))
            rect = (
                self.deviceTransform(view.viewportTransform().inverted()[0])
                .map(rect)
                .boundingRect()
            )

        rect.moveTo(self.anchorPos(self.anchor))

        if self.alignment & QtCore.Qt.AlignHCenter:
            rect.translate(-rect.width() / 2.0, 0)
        elif self.alignment & QtCore.Qt.AlignRight:
            rect.translate(-rect.width(), 0)

        if self.alignment & QtCore.Qt.AlignVCenter:
            rect.translate(0, -rect.height() / 2.0)
        elif self.alignment & QtCore.Qt.AlignBottom:
            rect.translate(0, -rect.height())

        return rect


class AnchoredLabelItem(AnchoredItem):
    """Draws a label with a black outline for increased visibility."""

    labelChanged = QtCore.Signal(str)

    def __init__(
        self,
        text: str,
        anchor: Union[QtCore.Qt.AnchorPoint, QtCore.Qt.Corner],
        alignment: QtCore.Qt.Alignment,
        parent: QtWidgets.QGraphicsItem,
        font: Optional[QtGui.QFont] = None,
        color: Optional[QtGui.QColor] = None,
    ):
        super().__init__(anchor, alignment, parent)

        if font is None:
            font = QtGui.QFont()
        if color is None:
            color = QtCore.Qt.white

        self._text = text
        self.font = font
        self.color = color

        self.action_edit_label = qAction(
            "edit-rename",
            "Rename",
            "Rename the current element.",
            self.editLabel,
        )

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text
        self.prepareGeometryChange()

    def unalignedRect(self):
        fm = QtGui.QFontMetrics(self.font)
        return QtCore.QRectF(0, 0, fm.boundingRect(self._text).width(), fm.height())

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        painter.save()
        painter.translate(self.alignedPos(self.anchorPos(self.anchor), self.alignment))

        fm = QtGui.QFontMetrics(self.font, painter.device())
        path = QtGui.QPainterPath()
        path.addText(0, fm.ascent(), self.font, self.text())

        painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.fillPath(path, QtGui.QBrush(self.color, QtCore.Qt.SolidPattern))
        painter.restore()

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.editLabel()
        super().mouseDoubleClickEvent(event)

    def editLabel(self) -> QtWidgets.QInputDialog:
        """Simple dialog for editing the label (and element name)."""
        dlg = QtWidgets.QInputDialog(self.scene().views()[0])
        dlg.setWindowTitle("Edit Name")
        dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
        dlg.setTextValue(self.text())
        dlg.setLabelText("Rename:")
        dlg.textValueSelected.connect(self.labelChanged)
        dlg.open()
        return dlg

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()
        menu.addAction(self.action_edit_label)
        menu.exec_(event.screenPos())
        event.accept()
