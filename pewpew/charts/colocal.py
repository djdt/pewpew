from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np

from pewpew.lib.numpyqt import array_to_polygonf
from pewpew.charts.base import BaseChart


class ColocalisationChart(BaseChart):
    def __init__(self, title: str = None, parent: QtWidgets.QWidget = None):
        super().__init__(QtCharts.QChart(), parent=parent)
        self.setMinimumSize(QtCore.QSize(640, 320))
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
        self.scatter.setColor(QtGui.QColor(0x8a, 0x3f, 0xfc))
        self.scatter.setUseOpenGL(True)
        self.scatter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.scatter.setMarkerSize(5)
        self.scatter.setMarkerShape(QtCharts.QScatterSeries.MarkerShapeRectangle)
        self.chart().addSeries(self.scatter)
        self.scatter.attachAxis(self.xaxis)
        self.scatter.attachAxis(self.yaxis)

        self.line = QtCharts.QLineSeries()
        self.line.setColor(QtGui.QColor(0xff, 0x7e, 0xb6))
        self.chart().addSeries(self.line)
        self.line.attachAxis(self.xaxis)
        self.line.attachAxis(self.yaxis)

        self.t1 = QtCharts.QLineSeries()
        self.t1.setColor(QtGui.QColor(0xff, 0xf1, 0xf1))
        self.chart().addSeries(self.t1)
        self.t1.attachAxis(self.xaxis)
        self.t1.attachAxis(self.yaxis)

        self.t2 = QtCharts.QLineSeries()
        self.t2.setColor(QtGui.QColor(0xff, 0xf1, 0xf1))
        self.chart().addSeries(self.t2)
        self.t2.attachAxis(self.xaxis)
        self.t2.attachAxis(self.yaxis)

    def drawPoints(self, x: np.ndarray, y: np.ndarray) -> None:
        points = np.stack([x.flat, y.flat], axis=1)
        poly = array_to_polygonf(points)
        self.scatter.replace(poly)

    def drawLine(self, a: float, b: float) -> None:
        x, y = (1.0, a + b) if a + b < 1.0 else ((1.0 - b) / a, 1.0)

        self.line.replace([QtCore.QPointF(0.0, b), QtCore.QPointF(1.0, y)])

    def drawThresholds(self, t1: float, t2: float) -> None:
        self.t1.replace([QtCore.QPointF(t1, 0.0), QtCore.QPointF(t1, 1.0)])
        self.t2.replace([QtCore.QPointF(0.0, t2), QtCore.QPointF(1.0, t2)])
