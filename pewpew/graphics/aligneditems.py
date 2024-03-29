from PySide6 import QtCore, QtGui, QtWidgets


class AlignedTextItem(QtWidgets.QGraphicsObject):
    """Draws a label with a black outline for increased visibility."""

    def __init__(
        self,
        parent: QtWidgets.QGraphicsItem,
        text: str,
        alignment: QtCore.Qt.Alignment | None = None,
        font: QtGui.QFont | None = None,
        brush: QtGui.QBrush | None = None,
        pen: QtGui.QPen | None = None,
    ):
        super().__init__(parent)

        if alignment is None:
            alignment = QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 2.0)
            pen.setCosmetic(True)

        self.alignment = alignment

        self._text = text
        self._font = font or QtGui.QFont()

        self.brush = brush or QtGui.QBrush(QtCore.Qt.white)
        self.pen = pen

    def font(self) -> QtGui.QFont:
        return self._font

    def setFont(self, font: QtGui.QFont) -> None:
        self._font = font

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text
        self.prepareGeometryChange()

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font())
        rect = QtCore.QRectF(0, 0, fm.boundingRect(self.text()).width(), fm.height())
        parent_rect = self.parentItem().boundingRect()

        if self.alignment & QtCore.Qt.AlignRight:
            rect.moveRight(parent_rect.right())
        elif self.alignment & QtCore.Qt.AlignHCenter:
            rect.moveCenter(QtCore.QPointF(parent_rect.center().x(), rect.center().y()))
        if self.alignment & QtCore.Qt.AlignBottom:
            rect.moveBottom(parent_rect.bottom())
        elif self.alignment & QtCore.Qt.AlignVCenter:
            rect.moveCenter(QtCore.QPointF(rect.center().x(), parent_rect.center().y()))
        return rect.intersected(parent_rect)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()

        painter.setClipRect(self.parentItem().boundingRect())
        painter.setFont(self.font())
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = painter.boundingRect(
            self.parentItem().boundingRect(), self.alignment, self.text()
        )

        path = QtGui.QPainterPath()
        path.addText(
            rect.left(),
            rect.top() + painter.fontMetrics().ascent(),
            painter.font(),
            self.text(),
        )
        painter.strokePath(path, self.pen)
        painter.fillPath(path, self.brush)

        painter.restore()


class UnscaledAlignedTextItem(AlignedTextItem):
    """AlignedTextItem that removes scaling from the view."""

    def __init__(
        self,
        parent: QtWidgets.QGraphicsItem,
        text: str,
        alignment: QtCore.Qt.Alignment | None = None,
        font: QtGui.QFont | None = None,
        brush: QtGui.QBrush | None = None,
        pen: QtGui.QPen | None = None,
    ):
        super().__init__(parent, text, alignment, font, brush, pen)

    def font(self) -> QtGui.QFont:
        font = QtGui.QFont(self._font)
        if self.scene() is not None:
            view = next(iter(self.scene().views()))
            font.setPointSizeF(font.pointSizeF() / view.transform().m22())
        return font
