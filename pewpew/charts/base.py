from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

colors = {
    "light": {
        "background": QtGui.QColor(0xFF, 0xFF, 0xFF),
        "axis": QtGui.QColor(0x8D, 0x8D, 0x8D),
        "grid": QtGui.QColor(0xE0, 0xE0, 0xE0),
        "title": QtGui.QColor(0x16, 0x16, 0x16),
        "text": QtGui.QColor(0x39, 0x39, 0x39),
        "colors": [QtGui.QColor(0x69, 0x29, 0xc4), QtGui.QColor(0x01, 0x27, 0x49)]
    },
    "dark": {
        "background": QtGui.QColor(0x16, 0x16, 0x16),
        "axis": QtGui.QColor(0x6F, 0x6F, 0x6F),
        "grid": QtGui.QColor(0x39, 0x39, 0x39),
        "title": QtGui.QColor(0xF4, 0xF4, 0xF4),
        "text": QtGui.QColor(0xC6, 0xC6, 0xC6),
    },
}


class BaseChart(QtCharts.QChartView):
    bgcolor = QtGui.QColor(0xF4, 0xF4, 0xF4)

    u1color = QtGui.QColor(0xFF, 0xFF, 0xFF)
    u2color = QtGui.QColor(0x39, 0x39, 0x39)

    d1color = QtGui.QColor(0x52, 0x52, 0x52)

    t1color = QtGui.QColor(0xF4, 0xF4, 0xF4)
    t2color = QtGui.QColor(0xC6, 0xC6, 0xC6)

    def __init__(
        self,
        chart: QtCharts.QChart,
        mode: str = "light",
        parent: QtWidgets.QWidget = None,
    ):
        self.mode = mode

        chart.setBackgroundBrush(QtGui.QBrush(colors[self.mode]["background"]))
        chart.setBackgroundPen(QtGui.QPen(colors[self.mode]["background"]))

        super().__init__(chart, parent)

    def addAxis(
        self, axis: QtCharts.QAbstractAxis, alignment: QtCore.Qt.Alignment
    ) -> None:
        axis.setMinorGridLineVisible(True)
        axis.setTitleBrush(QtGui.QBrush(colors[self.mode]["title"]))
        axis.setGridLinePen(QtGui.QPen(colors[self.mode]["grid"], 1.0))
        axis.setLinePen(QtGui.QPen(colors[self.mode]["axis"], 2.0))
        axis.setLabelsColor(colors[self.mode]["text"])
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
