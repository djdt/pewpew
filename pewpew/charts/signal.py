from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np

from pewpew.charts.base import BaseChart
from pewpew.charts.colors import light_theme, sequential, highlights

from pewpew.lib.numpyqt import array_to_polygonf


class SignalChart(BaseChart):
    def __init__(self, title: str = None, parent: QtWidgets.QWidget = None):
        super().__init__(QtCharts.QChart(), theme=light_theme, parent=parent)
        self.setRubberBand(QtCharts.QChartView.RectangleRubberBand)
        self.setMinimumSize(QtCore.QSize(640, 480))
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        if title is not None:
            self.chart().setTitle(title)

        self.chart().legend().hide()

        self.xaxis = QtCharts.QValueAxis()
        self.yaxis = QtCharts.QValueAxis()

        self.addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.signals = {}

    def setSignal(self, name: str, ys: np.ndarray) -> None:
        xs = np.arange(ys.size)
        data = np.stack((xs, ys), axis=1)
        poly = array_to_polygonf(data)
        self.signals[name].replace(poly)

    def addSignal(
        self, name: str, ys: np.ndarray, color: QtGui.QColor = QtCore.Qt.black
    ) -> None:
        series = QtCharts.QLineSeries()
        series.setPen(QtGui.QPen(color, 1.0))
        series.setUseOpenGL(True)  # Speed for many line?
        self.chart().addSeries(series)
        series.attachAxis(self.xaxis)
        series.attachAxis(self.yaxis)
        self.signals[name] = series

        self.setSignal(name, ys)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.RightButton:
            self.chart().zoomReset()
        else:
            super().mouseReleaseEvent(event)
        self.yaxis.applyNiceNumbers()
