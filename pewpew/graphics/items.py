from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from pewpew.actions import qAction
from pewpew.graphics.aligneditems import UnscaledAlignedTextItem

from typing import List, Optional


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
    editRequested = QtCore.Signal()
    mouseOverBar = QtCore.Signal(float)

    def __init__(
        self,
        parent: QtWidgets.QGraphicsItem,
        font: Optional[QtGui.QFont] = None,
        brush: Optional[QtGui.QBrush] = None,
        pen: Optional[QtGui.QPen] = None,
        # checkmarks: bool = False,
        orientation: Optional[QtCore.Qt.Orientation] = QtCore.Qt.Horizontal,
    ):
        super().__init__(parent)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 2.0)
            pen.setCosmetic(True)

        self._font = font or QtGui.QFont()
        self.brush = brush or QtGui.QBrush(QtCore.Qt.white)
        self.pen = pen
        assert orientation == QtCore.Qt.Horizontal
        self.orientation = orientation

        self.image = QtGui.QImage(
            np.arange(256, dtype=np.uint8), 256, 1, 256, QtGui.QImage.Format_Indexed8
        )

        self.vmin = 0.0
        self.vmax = 0.0
        self.unit = ""

    def font(self) -> QtGui.QFont:
        font = QtGui.QFont(self._font)
        if self.scene() is not None:
            view = next(iter(self.scene().views()))
            font.setPointSizeF(font.pointSizeF() / view.transform().m22())
        return font

    def updateTable(self, colortable: List[int], vmin: float, vmax: float, unit: str):
        self.vmin = vmin
        self.vmax = vmax
        self.unit = unit

        self.image.setColorTable(colortable)
        self.image.setColorCount(len(colortable))
        self.update()

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font())

        parent_rect = self.parentItem().boundingRect()
        if self.orientation == QtCore.Qt.Horizontal:
            rect = QtCore.QRectF(
                0,
                0,
                parent_rect.width(),
                fm.ascent() / 4.0 + fm.ascent() / 3.0 + fm.height(),
            )
        else:
            rect = QtCore.QRectF(
                0,
                0,
                fm.ascent() / 4.0 + fm.ascent() / 3.0 + fm.width("0000000"),
                parent_rect.height(),
            )
        return rect

    def niceTextValues(self, n: int = 7, trim: int = 0) -> np.ndarray:
        vrange = self.vmax - self.vmin
        interval = vrange / (n + 2 * trim)

        pwr = 10 ** int(np.log10(interval) - (1 if interval < 1.0 else 0))
        interval = interval / pwr

        idx = np.searchsorted(self.nicenums, interval)
        idx = min(idx, len(self.nicenums) - 1)

        interval = self.nicenums[idx] * pwr
        values = np.arange(int(self.vmin / interval) * interval, self.vmax, interval)
        return values[trim : values.size - trim]

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        painter.save()

        font = self.font()
        fm = QtGui.QFontMetrics(font, painter.device())
        rect = self.boundingRect()

        painter.drawImage(
            QtCore.QRectF(0, fm.ascent() / 4.0, rect.width(), fm.ascent() / 3.0),
            self.image,
        )

        path = QtGui.QPainterPath()
        path.addText(
            rect.width() - fm.boundingRect(self.unit).width() - fm.widthChar("m"),
            fm.ascent() * 0.25 + fm.height(),
            font,
            self.unit,
        )

        vrange = self.vmax - self.vmin
        if vrange <= 0.0:
            return

        for value in self.niceTextValues(7, trim=1):
            x = rect.width() * (value - self.vmin) / vrange
            text = f"{value:.6g}"
            path.addText(
                x - fm.boundingRect(text).width() / 2.0,
                fm.ascent() * 0.25 + fm.ascent() / 3.0 + fm.ascent(),
                font,
                text,
            )
            # if self.checkmarks:
            #     path.addRect(
            #         x - fm.lineWidth() / 2.0,
            #         fm.ascent() + fm.underlinePos(),
            #         fm.lineWidth() * 2.0,
            #         fm.underlinePos(),
            #     )
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.strokePath(path, self.pen)
        painter.fillPath(path, self.brush)
        painter.restore()

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.editRequested.emit()
        super().mouseDoubleClickEvent(event)


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
