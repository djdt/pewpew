from PySide6 import QtCore, QtGui, QtWidgets
from PySide6 import QtCharts

import numpy as np

from pewpew.charts.base import BaseChart, NiceValueAxis
from pewpew.charts.colors import light_theme

from pewpew.lib.numpyqt import array_to_polygonf


class SignalChart(BaseChart):
    """BaseChart for drawing raw LAICPMS signals.

    Series are stored in a dictionary variable 'series'.
    """

    def __init__(self, title: str = None, parent: QtWidgets.QWidget = None):
        super().__init__(
            QtCharts.QChart(), theme=light_theme, allow_navigation=True, parent=parent
        )
        self.setRubberBand(QtCharts.QChartView.RectangleRubberBand)
        self.setMinimumSize(QtCore.QSize(640, 480))
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        if title is not None:
            self.chart().setTitle(title)

        # self.chart().legend().hide()

        self.xaxis = QtCharts.QValueAxis()
        self.yaxis = NiceValueAxis()
        self.xaxis.setVisible(False)

        self.addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.series = {}

    def setSeries(self, name: str, ys: np.ndarray, xs: np.ndarray = None) -> None:
        if xs is None:
            xs = np.arange(ys.size)
        data = np.stack((xs, ys), axis=1)
        poly = array_to_polygonf(data)
        self.series[name].replace(poly)

    def addLineSeries(
        self,
        name: str,
        ys: np.ndarray,
        xs: np.ndarray = None,
        color: QtGui.QColor = QtCore.Qt.black,
        linewidth: float = 1.0,
        label: str = None,
    ) -> None:
        """Add a line plot to the chart.

        Args:
            name: key for series
            ys: y data
            xs: x data, defaults is range(ys.size)
            color: color of lines
            linewidth: width of lines
            label: optional label in legend
        """
        series = QtCharts.QLineSeries()
        self.chart().addSeries(series)
        series.setColor(color)
        series.setPen(QtGui.QPen(color, linewidth))
        series.setUseOpenGL(True)  # Speed for many line?

        if label is not None:
            series.setName(label)
        else:
            self.chart().legend().markers(series)[0].setVisible(False)

        series.attachAxis(self.xaxis)
        series.attachAxis(self.yaxis)
        self.series[name] = series

        self.setSeries(name, ys, xs=xs)

    def addScatterSeries(
        self,
        name: str,
        ys: np.ndarray,
        xs: np.ndarray = None,
        color: QtGui.QColor = QtCore.Qt.black,
        markersize: float = 10.0,
        label: str = None,
    ) -> None:
        """Add a scatter plot to the chart.

        Args:
            name: key for series
            ys: y data
            xs: x data, defaults is range(ys.size)
            color: color of markers
            markersize: size of markers
            label: optional label in legend
        """
        series = QtCharts.QScatterSeries()
        self.chart().addSeries(series)

        series.setColor(color)
        series.setPen(QtGui.QPen(color, 1.0))
        series.setBrush(QtGui.QBrush(color))
        series.setMarkerSize(markersize)
        series.setUseOpenGL(True)  # Speed for many line?

        if label is not None:
            series.setName(label)
        else:
            self.chart().legend().markers(series)[0].setVisible(False)

        series.attachAxis(self.xaxis)
        series.attachAxis(self.yaxis)
        self.series[name] = series

        self.setSeries(name, ys, xs=xs)
