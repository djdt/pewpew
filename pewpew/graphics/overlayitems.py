"""Items that can be used by the classes in overlaygraphics.

..Todo

Write MetricScaleBarOverlay as a GraphicsItem, handling text painting.
"""
from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np

from typing import List, Tuple


class LabelOverlay(QtWidgets.QGraphicsObject):
    editRequested = QtCore.Signal(str)
    """Draws the label with an outline for increased visibility."""

    def __init__(
        self,
        text: str,
        font: QtGui.QFont = None,
        color: QtGui.QColor = None,
        parent: QtWidgets.QGraphicsItem = None,
    ):
        super().__init__(parent)

        if font is None:
            font = QtGui.QFont()
        if color is None:
            color = QtCore.Qt.white

        self._text = text
        self.font = font
        self.color = color

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
        widget: QtWidgets.QWidget = None,
    ):

        fm = QtGui.QFontMetrics(self.font, painter.device())
        path = QtGui.QPainterPath()
        path.addText(0, fm.ascent(), self.font, self.text())

        painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.fillPath(path, QtGui.QBrush(self.color, QtCore.Qt.SolidPattern))

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.editRequested.emit(self.text())
        super().mouseDoubleClickEvent(event)


class ColorBarOverlay(QtWidgets.QGraphicsObject):
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
        font: QtGui.QFont = None,
        color: QtGui.QColor = None,
        checkmarks: bool = False,
        parent: QtWidgets.QGraphicsItem = None,
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
        widget: QtWidgets.QWidget = None,
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
        font: QtGui.QFont = None,
        color: QtGui.QColor = None,
        parent: QtWidgets.QGraphicsItem = None,
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
        widget: QtWidgets.QWidget = None,
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
