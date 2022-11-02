from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np

from pewpew.actions import qAction
from pewpew.graphics.aligneditems import UnscaledAlignedTextItem

from typing import List


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
        # checkmarks: bool = False,
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
        self.image = QtGui.QImage(self._data, 256, 1, 256, QtGui.QImage.Format_Indexed8)

        self.vmin = 0.0
        self.vmax = 0.0
        self.unit = ""

    # def font(self) -> QtGui.QFont:
    #     font = QtGui.QFont(self._font)
    #     if self.scene() is not None:
    #         view = next(iter(self.scene().views()))
    #         font.setPointSizeF(font.pointSizeF() / view.transform().m22())
    #     return font

    def updateTable(self, colortable: List[int], vmin: float, vmax: float, unit: str):
        self.vmin = vmin
        self.vmax = vmax
        self.unit = unit

        self.image.setColorTable(colortable)
        self.image.setColorCount(len(colortable))
        self.update()

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font)

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
            rect = QtCore.QRectF(0, 0, length, 5.0 + 10.0 + fm.height())
        else:
            raise NotImplementedError
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
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        fm = painter.fontMetrics()

        view = self.scene().views()[0]
        length = (
            view.mapFromScene(self.parentItem().boundingRect()).boundingRect().width()
        )

        painter.setFont(self.font)
        painter.rotate(self.parentItem().rotation())
        painter.rotate(self.rotation())

        painter.drawImage(
            QtCore.QRectF(0.0, 5.0, length, 10.0),
            self.image,
        )

        path = QtGui.QPainterPath()
        unit_pos = length - fm.boundingRect(self.unit).width() - fm.maxWidth()
        path.addText(
            unit_pos,
            5.0 + 10.0 + fm.height(),
            self.font,
            self.unit,
        )

        vrange = self.vmax - self.vmin
        if vrange <= 0.0:
            return

        px = 0.0  # Store previous text left pos
        for value in self.niceTextValues(7, trim=1):
            x = length * (value - self.vmin) / vrange
            text = f"{value:.6g}"
            half_width = fm.boundingRect(text).width()
            if (
                x - half_width > px and x + half_width < unit_pos
            ):  # Only add non-overlapping
                path.addText(
                    x - half_width,
                    5.0 + 10.0 + fm.height(),
                    self.font,
                    text,
                )
                px = x + half_width

        painter.setRenderHint(QtGui.QPainter.Antialiasing)
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
