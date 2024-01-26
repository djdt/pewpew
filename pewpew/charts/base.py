from PySide6 import QtCore, QtGui, QtWidgets
from PySide6 import QtCharts

import numpy as np



class NiceValueAxis(QtCharts.QValueAxis):
    """A chart axis that uses easy to read tick intervals.

    Uses *at least* 'nticks' ticks, may be more.
    """

    nicenums = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.5]

    def __init__(self, nticks: int = 6, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.nticks = nticks

        self.setLabelFormat("%.4g")
        self.setTickType(QtCharts.QValueAxis.TicksDynamic)
        self.setTickAnchor(0.0)
        self.setTickInterval(1e3)

    def setRange(self, amin: float, amax: float) -> None:
        self.fixValues(amin, amax)
        super().setRange(amin, amax)

    def fixValues(self, amin: float, amax: float) -> None:
        delta = amax - amin

        interval = delta / self.nticks
        pwr = 10 ** int(np.log10(interval) - (1 if interval < 1.0 else 0))
        interval = interval / pwr

        idx = np.searchsorted(NiceValueAxis.nicenums, interval)
        idx = min(idx, len(NiceValueAxis.nicenums) - 1)

        interval = NiceValueAxis.nicenums[idx] * pwr
        anchor = int(amin / interval) * interval

        self.setTickAnchor(anchor)
        self.setTickInterval(interval)


class BaseChart(QtCharts.QChartView):
    """QChartView with basic mouse naviagtion and styling.

    Valid keys for 'theme' are "background", "axis", "grid", "title", "text".
    A context menu implements copying the chart to the system clipboard and reseting the chart view.

    Args:
        chart: chart to display
        theme: dict of theme keys and colors
        allow_navigation: use mouse navigation
        parent: parent widget
    """

    def __init__(
        self,
        chart: QtCharts.QChart,
        theme: dict[str, QtGui.QColor],
        allow_navigation: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        self.theme = theme
        self.allow_navigation = allow_navigation

        self.nav_pos: QtCore.QPointF | None = None

        chart.setBackgroundBrush(QtGui.QBrush(self.theme["background"]))
        chart.setBackgroundPen(QtGui.QPen(self.theme["background"]))
        super().__init__(chart, parent)

    def addAxis(
        self, axis: QtCharts.QAbstractAxis, alignment: QtCore.Qt.Alignment
    ) -> None:
        axis.setMinorGridLineVisible(True)
        axis.setTitleBrush(QtGui.QBrush(self.theme["title"]))
        axis.setGridLinePen(QtGui.QPen(self.theme["grid"], 1.0))
        axis.setMinorGridLinePen(QtGui.QPen(self.theme["grid"], 0.5))
        axis.setLinePen(QtGui.QPen(self.theme["axis"], 1.0))
        axis.setLabelsColor(self.theme["text"])
        axis.setShadesColor(self.theme["title"])

        if isinstance(axis, QtCharts.QValueAxis):
            axis.setTickCount(6)
        self.chart().addAxis(axis, alignment)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton and self.allow_navigation:
            self.nav_pos = event.position()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.nav_pos is not None and self.allow_navigation:
            pos = event.position()
            offset = self.nav_pos - pos
            self.chart().scroll(offset.x(), -offset.y())
            self.nav_pos = event.position()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.nav_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.allow_navigation:
            scale = pow(2, event.angleDelta().y() / 360.0)
            self.chart().zoom(scale)
        super().wheelEvent(event)

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        action_copy_image = QtGui.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy To Clipboard", self
        )
        action_copy_image.setStatusTip("Copy the graphics view to the clipboard.")
        action_copy_image.triggered.connect(self.copyToClipboard)

        action_reset_zoom = QtGui.QAction(
            QtGui.QIcon.fromTheme("zoom-original"), "Reset Zoom", self
        )
        action_reset_zoom.setStatusTip("Reset the chart to the orignal view.")
        action_reset_zoom.triggered.connect(self.chart().zoomReset)

        menu = QtWidgets.QMenu(self.parent())
        menu.addAction(action_copy_image)
        menu.addAction(action_reset_zoom)
        menu.popup(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy image of the current chart to the system clipboard."""
        QtWidgets.QApplication.clipboard().setPixmap(self.grab(self.viewport().rect()))
