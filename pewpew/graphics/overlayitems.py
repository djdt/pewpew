from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np

from pewpew.actions import qAction

from typing import List, Optional, Tuple, Union


class OverlayItem(object):
    """Item to draw as an overlay.

    Overlay items are a fixed sized and are anchored to a position.

    Args:
        item: item to overlay
        anchor: side or corner to anchor item
        alignment: how to align item relative to anchor
    """

    def __init__(
        self,
        item: QtWidgets.QGraphicsItem,
        anchor: Optional[Union[QtCore.Qt.AnchorPoint, QtCore.Qt.Corner]] = None,
        alignment: Optional[QtCore.Qt.Alignment] = None,
    ):
        if anchor is None:
            anchor = QtCore.Qt.TopLeftCorner
        if alignment is None:
            alignment = QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft

        self.item = item
        self.anchor = anchor
        self.alignment = alignment

    def anchorPos(self, rect: QtCore.QRectF) -> QtCore.QPointF:
        if isinstance(self.anchor, QtCore.Qt.Corner):
            if self.anchor == QtCore.Qt.TopLeftCorner:
                pos = rect.topLeft()
            elif self.anchor == QtCore.Qt.TopRightCorner:
                pos = rect.topRight()
            elif self.anchor == QtCore.Qt.BottomLeftCorner:
                pos = rect.bottomLeft()
            else:  # BottomRightCorner
                pos = rect.bottomRight()
        else:  # AnchorPoint
            if self.anchor == QtCore.Qt.AnchorTop:
                pos = QtCore.QPointF(rect.center().x(), rect.top())
            elif self.anchor == QtCore.Qt.AnchorLeft:
                pos = QtCore.QPointF(rect.left(), rect.center().y())
            elif self.anchor == QtCore.Qt.AnchorRight:
                pos = QtCore.QPointF(rect.right(), rect.center().y())
            elif self.anchor == QtCore.Qt.AnchorBottom:
                pos = QtCore.QPointF(rect.center().x(), rect.bottom())
            else:
                raise ValueError("Only Top, Left, Right, Bottom anchors supported.")

        return pos

    def pos(self) -> QtCore.QPointF:
        pos = self.item.pos()  # Aligned Left and Top
        rect = self.item.boundingRect()

        if self.alignment & QtCore.Qt.AlignHCenter:
            pos.setX(pos.x() - rect.width() / 2.0)
        elif self.alignment & QtCore.Qt.AlignRight:
            pos.setX(pos.x() - rect.width())

        if self.alignment & QtCore.Qt.AlignVCenter:
            pos.setY(pos.y() - rect.height() / 2.0)
        elif self.alignment & QtCore.Qt.AlignBottom:
            pos.setY(pos.y() - rect.height())

        return pos

    def contains(self, view_pos: QtCore.QPoint, view_rect: QtCore.QRect) -> bool:
        rect = self.item.boundingRect()
        rect.moveTo(self.pos() + self.anchorPos(view_rect))
        return rect.contains(view_pos)


class LabelOverlay(QtWidgets.QGraphicsObject):
    """Draws a label with a black outline for increased visibility."""

    labelChanged = QtCore.Signal(str)

    def __init__(
        self,
        text: str,
        font: Optional[QtGui.QFont] = None,
        color: Optional[QtGui.QColor] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent)

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

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font)
        return QtCore.QRectF(0, 0, fm.boundingRect(self._text).width(), fm.height())

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):

        fm = QtGui.QFontMetrics(self.font, painter.device())
        path = QtGui.QPainterPath()
        path.addText(0, fm.ascent(), self.font, self.text())

        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.fillPath(path, QtGui.QBrush(self.color, QtCore.Qt.SolidPattern))

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


class ColorBarOverlay(QtWidgets.QGraphicsObject):
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
        colortable: List[int],
        vmin: float,
        vmax: float,
        unit: str = "",
        height: int = 16,
        font: Optional[QtGui.QFont] = None,
        color: Optional[QtGui.QColor] = None,
        checkmarks: bool = False,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent)

        if font is None:
            font = QtGui.QFont()
            font.setPointSize(16)

        image = QtGui.QImage(
            np.arange(256, dtype=np.uint8), 256, 1, 256, QtGui.QImage.Format_Indexed8
        )
        image.setColorTable(colortable)
        self.pixmap = QtGui.QPixmap.fromImage(image)

        self.vmin = vmin
        self.vmax = vmax
        self.unit = unit
        self.height = height

        if font is None:
            font = QtGui.QFont()
        if color is None:
            color = QtCore.Qt.white

        self.font = font
        self.color = color
        self.checkmarks = checkmarks

    def updateTable(self, colortable: List[int], vmin: float, vmax: float):
        self.vmin = vmin
        self.vmax = vmax

        image = QtGui.QImage(
            np.arange(256, dtype=np.uint8), 256, 1, 256, QtGui.QImage.Format_Indexed8
        )
        image.setColorTable(colortable)
        self.pixmap = QtGui.QPixmap.fromImage(image)

    def boundingRect(self) -> QtCore.QRectF:
        fm = QtGui.QFontMetrics(self.font)
        if self.scene() is None:
            return QtCore.QRectF(0, 0, 100, self.height + fm.height())

        view = next(iter(self.scene().views()))
        rect = QtCore.QRectF(0, 0, view.viewport().width(), self.height + fm.height())
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
        width = painter.viewport().width()

        fm = QtGui.QFontMetrics(self.font, painter.device())

        rect = QtCore.QRect(0, fm.height(), width, self.height)
        painter.drawPixmap(rect, self.pixmap)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.white, QtCore.Qt.NoBrush))
        painter.drawRect(rect)

        path = QtGui.QPainterPath()
        # Todo: find a better pad value
        path.addText(
            width - fm.boundingRect(self.unit).width() - 10,
            fm.ascent(),
            self.font,
            self.unit,
        )

        vrange = self.vmax - self.vmin
        if vrange <= 0.0:
            return

        for value in self.niceTextValues(7, trim=1):
            x = width * (value - self.vmin) / vrange
            text = f"{value:.6g}"
            path.addText(
                x - fm.boundingRect(text).width() / 2.0,
                fm.ascent(),
                self.font,
                text,
            )
            if self.checkmarks:
                path.addRect(
                    x - fm.lineWidth() / 2.0,
                    fm.ascent() + fm.underlinePos(),
                    fm.lineWidth() * 2.0,
                    fm.underlinePos(),
                )
        painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.fillPath(path, QtGui.QBrush(self.color, QtCore.Qt.SolidPattern))

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.editRequested.emit()
        super().mouseDoubleClickEvent(event)


class MetricScaleBarOverlay(QtWidgets.QGraphicsItem):
    """Draw a scalebar.

    Uses metric units with a base of 1 pixel = 1 μm.

    Args:
        width: maxmium width in px
        height: height in px
        font: label font
        color: font color
        parent: parent item
    """

    allowed_lengths = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0]
    units = {
        "pm": 1e-12,
        "nm": 1e-9,
        "μm": 1e-6,
        "mm": 1e-3,
        "cm": 1e-2,
        "m": 1.0,
        "km": 1e3,
    }

    def __init__(
        self,
        width: int = 200,
        height: int = 10,
        font: Optional[QtGui.QFont] = None,
        color: Optional[QtGui.QColor] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent)

        self.unit = "μm"
        self.width = width
        self.height = height

        if font is None:
            font = QtGui.QFont()
        if color is None:
            color = QtCore.Qt.white

        self.font = font
        self.color = color

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font)
        return QtCore.QRectF(0, 0, self.width, self.height + fm.height())

    def getWidthAndUnit(self, desired: float) -> Tuple[float, str]:
        base = desired * self.units[self.unit]

        units = list(self.units.keys())
        factors = list(self.units.values())
        idx = np.max(np.searchsorted(factors, base) - 1, 0)

        new = self.allowed_lengths[
            np.searchsorted(self.allowed_lengths, base / factors[idx]) - 1
        ]
        new_unit = units[idx]

        return new * factors[idx] / self.units[self.unit], new_unit

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        view = self.scene().views()[0]
        view_width = view.mapToScene(0, 0, self.width, 1).boundingRect().width()
        width, unit = self.getWidthAndUnit(view_width)
        # Current scale
        text = f"{width * self.units[self.unit] / self.units[unit]:.3g} {unit}"
        width = width * view.transform().m11()

        fm = QtGui.QFontMetrics(self.font, painter.device())
        path = QtGui.QPainterPath()
        path.addText(
            self.width / 2.0 - fm.boundingRect(text).width() / 2.0,
            fm.ascent(),
            self.font,
            text,
        )

        painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.fillPath(path, QtGui.QBrush(self.color, QtCore.Qt.SolidPattern))

        # Draw the bar
        rect = QtCore.QRectF(
            self.width / 2.0 - width / 2.0, fm.height(), width, self.height
        )
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 1.0))
        painter.setBrush(QtGui.QBrush(self.color, QtCore.Qt.SolidPattern))
        painter.drawRect(rect)
