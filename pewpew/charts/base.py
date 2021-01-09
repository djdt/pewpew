from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

from typing import Dict


class BaseChart(QtCharts.QChartView):
    def __init__(
        self,
        chart: QtCharts.QChart,
        theme: Dict[str, QtGui.QColor],
        parent: QtWidgets.QWidget = None,
    ):
        self.theme = theme

        chart.setBackgroundBrush(QtGui.QBrush(self.theme["background"]))
        chart.setBackgroundPen(QtGui.QPen(self.theme["background"]))
        super().__init__(chart, parent)

    def addAxis(
        self, axis: QtCharts.QAbstractAxis, alignment: QtCore.Qt.Alignment
    ) -> None:
        axis.setMinorGridLineVisible(True)
        axis.setTitleBrush(QtGui.QBrush(self.theme["title"]))
        axis.setGridLinePen(QtGui.QPen(self.theme["grid"], 1.0))
        axis.setLinePen(QtGui.QPen(self.theme["axis"], 1.0))
        axis.setLabelsColor(self.theme["text"])
        if isinstance(axis, QtCharts.QValueAxis):
            axis.setTickCount(6)
        # axis.setShadesColor(colors[self.mode]["title"])
        self.chart().addAxis(axis, alignment)

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        action_copy_image = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy To Clipboard", self
        )
        action_copy_image.setStatusTip("Copy the graphics view to the clipboard.")
        action_copy_image.triggered.connect(self.copyToClipboard)

        action_reset_zoom = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("zoom-original"), "Reset Zoom", self
        )
        action_reset_zoom.setStatusTip("Reset the chart to the orignal view.")
        action_reset_zoom.triggered.connect(self.chart().zoomReset)

        context_menu = QtWidgets.QMenu(self.parent())
        context_menu.addAction(action_copy_image)
        context_menu.addAction(action_reset_zoom)
        context_menu.popup(event.globalPos())

    def copyToClipboard(self) -> None:
        QtWidgets.QApplication.clipboard().setPixmap(self.grab(self.viewport().rect()))
