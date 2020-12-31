from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np

from pewpew.charts.basicchart import BasicChartView

from pewpew.lib.numpyqt import array_to_polygonf


class BasicChart(QtCharts.QChartView):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.xaxis = QtCharts.QValueAxis()
        self.yaxis = QtCharts.QValueAxis()

        self.chart().addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.chart().addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.series = QtCharts.QScatterSeries()
        self.chart().addSeries(self.series)
        self.series.attachAxis(self.xaxis)
        self.series.attachAxis(self.yaxis)

    def setPoints(self, points: np.ndarray) -> None:
        if not (points.ndim == 2 and points.shape[1] == 2):
            raise ValueError("points must have shape (n, 2).")

        self.xaxis.setRange(np.amin(points[:, 0]), np.amax(points[:, 0]))
        self.yaxis.setRange(np.amin(points[:, 1]), np.amax(points[:, 1]))

        poly = array_to_polygonf(points)
        self.series.replace(poly)

        self.xaxis.applyNiceNumbers()
        self.yaxis.applyNiceNumbers()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        super().mouseReleaseEvent(event)
        self.xaxis.applyNiceNumbers()
        self.yaxis.applyNiceNumbers()
