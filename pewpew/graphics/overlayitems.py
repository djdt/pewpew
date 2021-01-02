"""Items that can be used by the classes in overlaygraphics.

..Todo

Write MetricScaleBarOverlay as a GraphicsItem, handling text painting.
"""
from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np

from typing import Tuple


class LabelOverlay(QtWidgets.QGraphicsItem):
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

        self.text = text
        self.font = font
        self.color = color

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font)
        return QtCore.QRectF(0, 0, fm.width(self.text), fm.height())

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):

        fm = QtGui.QFontMetrics(self.font, painter.device())
        path = QtGui.QPainterPath()
        path.addText(0, fm.ascent(), self.font, self.text)

        painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.fillPath(path, QtGui.QBrush(self.color, QtCore.Qt.SolidPattern))


class ColorBarOverlay(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        colortable: np.ndarray,
        vmin: float,
        vmax: float,
        unit: str = "",
        height: int = 16,
        font: QtGui.QFont = None,
        color: QtGui.QColor = None,
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

    def updateTable(self, colortable: np.ndarray, vmin: float, vmax: float):
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

        view = self.scene().views()[0]
        rect = QtCore.QRectF(0, 0, view.viewport().width(), self.height + fm.height())
        return rect

    def formatValue(self, x: float, sf: int = 2) -> float:
        x = np.asarray(x)
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (sf - 1))
        mags = 10 ** (sf - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget = None,
    ):
        view = self.scene().views()[0]
        width = view.viewport().width()

        path = QtGui.QPainterPath()
        fm = QtGui.QFontMetrics(self.font, painter.device())
        # Todo: find a better pad value
        path.addText(
            width - fm.width(self.unit) - 10, fm.ascent(), self.font, self.unit
        )
        for value in np.linspace(self.vmin, self.vmax, 7)[1:-1]:
            value = self.formatValue(value, 2)
            x = width * value / (self.vmax - self.vmin)
            text = f"{value:.6g}"
            path.addText(
                x - fm.width(text) / 2.0,
                fm.ascent(),
                self.font,
                text,
            )

        painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.fillPath(path, QtGui.QBrush(self.color, QtCore.Qt.SolidPattern))

        rect = QtCore.QRect(0, fm.height(), width, self.height)
        painter.drawPixmap(rect, self.pixmap)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.white, QtCore.Qt.NoBrush))
        painter.drawRect(rect)


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
            self.width / 2.0 - fm.width(text) / 2.0, fm.ascent(), self.font, text
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
