from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np

from pewpew.charts.base import BaseChart
from pewpew.charts.colors import light_theme, sequential

from pewpew.lib.numpyqt import array_to_polygonf


class SignalChart(BaseChart):
    def __init__(self, title: str = None, parent: QtWidgets.QWidget = None):
        super().__init__(
            QtCharts.QChart(), theme=light_theme, allow_navigation=True, parent=parent
        )
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

        self.series = {}

    def setSeries(self, name: str, ys: np.ndarray, xs: np.ndarray = None) -> None:
        if xs is None:
            xs = np.arange(ys.size)
        data = np.stack((xs, ys), axis=1)
        poly = array_to_polygonf(data)
        self.series[name].replace(poly)

    def addSeries(
        self,
        name: str,
        ys: np.ndarray,
        xs: np.ndarray = None,
        series_type: QtCharts.QAbstractSeries = QtCharts.QLineSeries,
        color: QtGui.QColor = QtCore.Qt.black,
    ) -> None:
        series = series_type()
        series.setColor(color)
        series.setPen(QtGui.QPen(color, 1.0))
        series.setUseOpenGL(True)  # Speed for many line?
        self.chart().addSeries(series)
        series.attachAxis(self.xaxis)
        series.attachAxis(self.yaxis)
        self.series[name] = series

        self.setSeries(name, ys, xs=xs)
