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

        self.signal = QtCharts.QLineSeries()
        self.signal.setPen(QtGui.QPen(QtCore.Qt.black, 1.0))
        self.signal.setUseOpenGL(True)  # Speed for many line?

        self.chart().addSeries(self.signal)
        self.signal.attachAxis(self.xaxis)
        self.signal.attachAxis(self.yaxis)

    def setSignal(self, ys: np.ndarray) -> None:
        xs = np.arange(ys.size)
        self.yvalues = ys
        data = np.stack((xs, ys), axis=1)
        poly = array_to_polygonf(data)
        self.signal.replace(poly)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.RightButton:
            self.chart().zoomReset()
        else:
            super().mouseReleaseEvent(event)
        self.xaxis.applyNiceNumbers()
        self.yaxis.applyNiceNumbers()
