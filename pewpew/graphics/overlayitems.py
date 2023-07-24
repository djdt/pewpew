from typing import List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.graphics.overlaygraphics import OverlayItem


class ColorBarOverlay(OverlayItem):
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
        font: QtGui.QFont | None = None,
        color: QtGui.QColor | None = None,
        checkmarks: bool = False,
        alignment: QtCore.Qt.AlignmentFlag | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(parent)

        if font is None:
            font = QtGui.QFont()
            font.setPointSize(16)
        if alignment is None:
            alignment = QtCore.Qt.AlignBottom

        image = QtGui.QImage(
            np.arange(256, dtype=np.uint8), 256, 1, 256, QtGui.QImage.Format_Indexed8
        )
        image.setColorTable(colortable)
        self.pixmap = QtGui.QPixmap.fromImage(image)

        self.vmin = vmin
        self.vmax = vmax
        self.unit = unit
        self.height = height

        self.font = font or QtGui.QFont()
        self.color = color or QtCore.Qt.white
        self.checkmarks = checkmarks
        self.alignment = alignment

    def updateTable(self, colortable: List[int], vmin: float, vmax: float, unit: str):
        self.vmin = vmin
        self.vmax = vmax
        self.unit = unit

        image = QtGui.QImage(
            np.arange(256, dtype=np.uint8), 256, 1, 256, QtGui.QImage.Format_Indexed8
        )
        image.setColorTable(colortable)
        self.pixmap = QtGui.QPixmap.fromImage(image)
        self.requestPaint()

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font)

        rect = QtCore.QRectF(0, 0, self.viewport.width(), self.height + fm.height())

        if self.alignment & QtCore.Qt.AlignRight:
            rect.moveRight(self.viewport.right())
        elif self.alignment & QtCore.Qt.AlignHCenter:
            rect.moveCenter(
                QtCore.QPointF(self.viewport.center().x(), rect.center().y())
            )
        if self.alignment & QtCore.Qt.AlignBottom:
            rect.moveBottom(self.viewport.bottom())
        elif self.alignment & QtCore.Qt.AlignVCenter:
            rect.moveCenter(
                QtCore.QPointF(rect.center().x(), self.viewport.center().y())
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
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()

        painter.translate(self.boundingRect().topLeft())

        fm = QtGui.QFontMetrics(self.font, painter.device())

        rect = QtCore.QRect(0, fm.height(), self.viewport.width(), self.height)
        painter.drawPixmap(rect, self.pixmap)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.white, QtCore.Qt.NoBrush))
        painter.drawRect(rect)

        path = QtGui.QPainterPath()
        # Todo: find a better pad value
        path.addText(
            self.viewport.width() - fm.boundingRect(self.unit).width() - 10,
            fm.ascent(),
            self.font,
            self.unit,
        )

        vrange = self.vmax - self.vmin
        if vrange <= 0.0:
            return

        for value in self.niceTextValues(7, trim=1):
            x = self.viewport.width() * (value - self.vmin) / vrange
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
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.fillPath(path, QtGui.QBrush(self.color, QtCore.Qt.SolidPattern))
        painter.restore()

        super().paint(painter, option, widget)

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.editRequested.emit()
        super().mouseDoubleClickEvent(event)


class MetricScaleBarOverlay(OverlayItem):
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
        font: QtGui.QFont | None = None,
        color: QtGui.QColor | None = None,
        alignment: QtCore.Qt.AlignmentFlag | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(parent)

        self.unit = "μm"
        self.width = width
        self.height = height

        if font is None:
            font = QtGui.QFont()
        if alignment is None:
            alignment = QtCore.Qt.AlignTop | QtCore.Qt.AlignRight

        self.font = font
        self.color = color or QtCore.Qt.white
        self.alignment = alignment

    def boundingRect(self):
        fm = QtGui.QFontMetrics(self.font)
        rect = QtCore.QRectF(0, 0, self.width, self.height + fm.height())

        if self.alignment & QtCore.Qt.AlignRight:
            rect.moveRight(self.viewport.right())
        elif self.alignment & QtCore.Qt.AlignHCenter:
            rect.moveCenter(
                QtCore.QPointF(self.viewport.center().x(), rect.center().y())
            )
        if self.alignment & QtCore.Qt.AlignBottom:
            rect.moveBottom(self.viewport.bottom())
        elif self.alignment & QtCore.Qt.AlignVCenter:
            rect.moveCenter(
                QtCore.QPointF(rect.center().x(), self.viewport.center().y())
            )
        return rect

    @staticmethod
    def getWidthAndUnit(desired: float, unit: str) -> Tuple[float, str]:
        base = desired * MetricScaleBarOverlay.units[unit]

        units = list(MetricScaleBarOverlay.units.keys())
        factors = list(MetricScaleBarOverlay.units.values())
        idx = np.max(np.searchsorted(factors, base) - 1, 0)

        new = MetricScaleBarOverlay.allowed_lengths[
            np.searchsorted(MetricScaleBarOverlay.allowed_lengths, base / factors[idx])
            - 1
        ]
        new_unit = units[idx]

        return new * factors[idx] / MetricScaleBarOverlay.units[unit], new_unit

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(self.boundingRect().topLeft())

        scale = self.parentItem().boundingRect().width() / self.viewport.width()
        width, unit = self.getWidthAndUnit(self.width * scale, self.unit)

        # Current scale
        text = f"{width * self.units[self.unit] / self.units[unit]:.3g} {unit}"
        width = width / scale

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
        painter.restore()

        super().paint(painter, option, widget)
