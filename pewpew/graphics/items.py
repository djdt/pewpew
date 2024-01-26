
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction
from pewpew.graphics.aligneditems import UnscaledAlignedTextItem
from pewpew.graphics.util import path_for_colorbar_labels


class ColorBarItem(QtWidgets.QGraphicsObject):
    """Draw a colorbar.

    Ticks are formatted at easily readable intervals.

    Args:
        colortable: the colortable to use
        vmin: minimum value
        vmax: maxmium value
        unit: also display a unit
        height: height of bar in pixels
        font: label font
        color: font color
        checkmarks: mark intervals
        parent: parent item
    """

    nicenums = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5]
    # editRequested = QtCore.Signal()
    # mouseOverBar = QtCore.Signal(float)

    def __init__(
        self,
        parent: QtWidgets.QGraphicsItem,
        font: QtGui.QFont | None = None,
        brush: QtGui.QBrush | None = None,
        pen: QtGui.QPen | None = None,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
    ):
        super().__init__(parent)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 2.0)
            pen.setCosmetic(True)

        self.font = font or QtGui.QFont()
        self.brush = brush or QtGui.QBrush(QtCore.Qt.white)
        self.pen = pen
        assert orientation == QtCore.Qt.Horizontal
        self.orientation = orientation

        self._data = np.arange(256, dtype=np.uint8)  # save a reference
        self._data[0] = 1
        self.image = QtGui.QImage(self._data, 256, 1, 256, QtGui.QImage.Format_Indexed8)

        self.vmin = 0.0
        self.vmax = 0.0
        self.unit = ""

    def updateTable(self, colortable: list[int], vmin: float, vmax: float, unit: str):
        self.vmin = vmin
        self.vmax = vmax
        self.unit = unit

        self.image.setColorTable(colortable)
        self.image.setColorCount(len(colortable))
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font)
        xh = fm.xHeight()

        if self.scene() is not None:
            view = self.scene().views()[0]
            length = (
                view.mapFromScene(self.parentItem().boundingRect())
                .boundingRect()
                .width()
            )
        else:
            length = 1.0

        if self.orientation == QtCore.Qt.Horizontal:
            height = xh * 2.0 + fm.height()
            if self.unit is not None and self.unit != "":
                height += fm.height()
            rect = QtCore.QRectF(0.0, 0.0, length, height)
        else:
            raise NotImplementedError
        return rect

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        painter.setFont(self.font)
        fm = painter.fontMetrics()
        xh = fm.xHeight()

        view = self.scene().views()[0]
        length = (
            view.mapFromScene(self.parentItem().boundingRect()).boundingRect().width()
        )

        rect = QtCore.QRectF(0.0, xh / 2.0, length, xh)

        painter.setFont(self.font)
        painter.rotate(self.parentItem().rotation())
        painter.rotate(self.rotation())

        painter.drawImage(rect, self.image)
        path = path_for_colorbar_labels(self.font, self.vmin, self.vmax, length)
        clip = QtCore.QRectF(0.0, 0.0, length, xh * 2.0 + fm.height())

        if self.unit is not None and self.unit != "":
            path.addText(
                rect.width()
                - fm.boundingRect(self.unit).width()
                - fm.lineWidth()
                - fm.rightBearing(self.unit[-1]),
                fm.ascent() + fm.height(),
                painter.font(),
                self.unit,
            )
            clip.setHeight(clip.height() + fm.height())

        path.translate(rect.bottomLeft())

        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setClipRect(clip)
        painter.strokePath(path, self.pen)
        painter.fillPath(path, self.brush)
        painter.restore()

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        # action_edit = qAction(
        #     "format-number-percent",
        #     "Set Range",
        #     "Set the numerical range of the colortable.",
        #     self.editRequested,
        # )
        action_hide = qAction(
            "visibility",
            "Hide Colorbar",
            "Hide the colortable scale bar.",
            self.hide,
        )
        menu = QtWidgets.QMenu()
        # menu.addAction(action_edit)
        menu.addAction(action_hide)
        menu.exec_(event.screenPos())
        event.accept()


class EditableLabelItem(UnscaledAlignedTextItem):
    labelChanged = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtWidgets.QGraphicsItem,
        text: str,
        label_text: str,
        alignment: QtCore.Qt.Alignment | None = None,
        font: QtGui.QFont | None = None,
        brush: QtGui.QBrush | None = None,
        pen: QtGui.QPen | None = None,
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


class RGBLabelItem(QtWidgets.QGraphicsObject):
    def __init__(
        self,
        parent: QtWidgets.QGraphicsItem,
        texts: list[str],
        colors: list[QtGui.QColor],
        alignment: QtCore.Qt.Alignment | None = None,
        font: QtGui.QFont | None = None,
        pen: QtGui.QPen | None = None,
    ):
        super().__init__(parent)

        if alignment is None:
            alignment = QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 2.0)
            pen.setCosmetic(True)

        self.alignment = alignment

        self._texts = texts
        self._font = font or QtGui.QFont()
        self.colors = colors

        self.pen = pen

    def font(self) -> QtGui.QFont:
        font = QtGui.QFont(self._font)
        if self.scene() is not None:
            view = next(iter(self.scene().views()))
            font.setPointSizeF(font.pointSizeF() / view.transform().m22())
        return font

    def setFont(self, font: QtGui.QFont) -> None:
        self._font = font

    def texts(self) -> list[str]:
        return self._texts

    def setTexts(self, texts: list[str]) -> None:
        self._texts = texts
        self.prepareGeometryChange()

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font())
        if len(self.texts()) > 0:
            width = max(fm.boundingRect(text).width() for text in self.texts())
        else:
            width = 0
        rect = QtCore.QRectF(0, 0, width, fm.height() * len(self.texts()))
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

        pos = self.boundingRect().topLeft()
        for text, color in zip(self.texts(), self.colors):
            path = QtGui.QPainterPath()
            path.addText(
                pos.x(), pos.y() + painter.fontMetrics().ascent(), painter.font(), text
            )
            painter.strokePath(path, self.pen)
            painter.fillPath(path, QtGui.QBrush(color))
            pos.setY(pos.y() + painter.fontMetrics().height())

        painter.restore()


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
        parent: QtWidgets.QGraphicsItem | None = None,
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

        self.selected_edge: str | None = None
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
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        painter.setPen(self.pen)
        painter.setBrush(self.brush)
        painter.drawRect(self.rect)
        painter.restore()

    def edgeAt(self, pos: QtCore.QPointF) -> str | None:
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
