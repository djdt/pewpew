from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np

from pewpew.charts.base import BaseChart
from pewpew.charts.colors import light_theme, sequential

from pewpew.lib.numpyqt import array_to_polygonf

from typing import Optional


class ColocalisationChart(BaseChart):
    """BaseChart for displaying a scatter plot of two arrays."""

    def __init__(self, title: Optional[str] = None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(QtCharts.QChart(), theme=light_theme, parent=parent)
        self.setRubberBand(QtCharts.QChartView.RectangleRubberBand)
        self.setMinimumSize(QtCore.QSize(640, 480))
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        if title is not None:
            self.chart().setTitle(title)

        self.chart().legend().hide()

        self.xaxis = QtCharts.QValueAxis()
        self.xaxis.setRange(0.0, 1.0)
        self.yaxis = QtCharts.QValueAxis()
        self.yaxis.setRange(0.0, 1.0)
        self.addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.scatter = QtCharts.QScatterSeries()
        self.scatter.setColor(sequential[1])
        self.scatter.setMarkerSize(5)
        self.scatter.setUseOpenGL(True)
        self.chart().addSeries(self.scatter)
        self.scatter.attachAxis(self.xaxis)
        self.scatter.attachAxis(self.yaxis)

        self.line = QtCharts.QLineSeries()
        self.line.setPen(QtGui.QPen(QtCore.Qt.black, 1.5))
        self.chart().addSeries(self.line)
        self.line.attachAxis(self.xaxis)
        self.line.attachAxis(self.yaxis)

        self.t1 = QtCharts.QLineSeries()
        self.t1.setPen(QtGui.QPen(QtCore.Qt.black, 1.0, QtCore.Qt.DashLine))
        self.chart().addSeries(self.t1)
        self.t1.attachAxis(self.xaxis)
        self.t1.attachAxis(self.yaxis)

        self.t2 = QtCharts.QLineSeries()
        self.t2.setPen(QtGui.QPen(QtCore.Qt.black, 1.0, QtCore.Qt.DashLine))
        self.chart().addSeries(self.t2)
        self.t2.attachAxis(self.xaxis)
        self.t2.attachAxis(self.yaxis)

    def drawPoints(self, x: np.ndarray, y: np.ndarray) -> None:
        """Plot scatter of 'x' and 'y'."""
        points = np.stack([x.flat, y.flat], axis=1)
        poly = array_to_polygonf(points)
        self.scatter.replace(poly)

    def drawLine(self, a: float, b: float) -> None:
        """Plot a line with gradient 'a' and intercept 'b'."""
        self.line.replace([QtCore.QPointF(b, 0.0), QtCore.QPointF(1.0, a + b)])

    def drawThresholds(self, t1: float, t2: float) -> None:
        """Draw horizontal and vertical lines at 't1' and 't2'."""
        self.t1.replace([QtCore.QPointF(t1, 0.0), QtCore.QPointF(t1, 1.0)])
        self.t2.replace([QtCore.QPointF(0.0, t2), QtCore.QPointF(1.0, t2)])
