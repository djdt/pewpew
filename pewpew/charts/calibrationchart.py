from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np

from pewpew.lib.numpyqt import array_to_polygonf


class CalibrationChart(QtCharts.QChartView):
    def __init__(self, title: str = None, parent: QtWidgets.QWidget = None):
        super().__init__(QtCharts.QChart(), parent)
        self.setRubberBand(QtCharts.QChartView.RectangleRubberBand)
        self.setMinimumSize(QtCore.QSize(640, 480))
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        if title is not None:
            self.chart().setTitle(title)

        self.chart().legend().hide()

        self.xaxis = QtCharts.QValueAxis()
        self.yaxis = QtCharts.QValueAxis()

        self.chart().addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.chart().addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.label_series = QtCharts.QScatterSeries()
        self.label_series.append(0, 0)
        self.label_series.setBrush(QtGui.QBrush(QtCore.Qt.black, QtCore.Qt.NoBrush))
        self.label_series.setPointLabelsFormat("(@xPoint, @yPoint)")

        self.chart().addSeries(self.label_series)
        self.label_series.attachAxis(self.xaxis)
        self.label_series.attachAxis(self.yaxis)

        self.line = QtCharts.QLineSeries()
        self.line.setColor(QtCore.Qt.red)

        self.chart().addSeries(self.line)
        self.line.attachAxis(self.xaxis)
        self.line.attachAxis(self.yaxis)

        self.series = QtCharts.QScatterSeries()
        self.series.setColor(QtCore.Qt.black)
        self.series.setMarkerSize(12)

        self.chart().addSeries(self.series)
        self.series.attachAxis(self.xaxis)
        self.series.attachAxis(self.yaxis)

        self.series.hovered.connect(self.showPointPosition)

    def setPoints(self, points: np.ndarray) -> None:
        if not (points.ndim == 2 and points.shape[1] == 2):
            raise ValueError("points must have shape (n, 2).")

        xmin, xmax = np.amin(points[:, 0]), np.amax(points[:, 0])
        ymin, ymax = np.amin(points[:, 1]), np.amax(points[:, 1])
        # self.xaxis.setRange(xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.05)
        # self.yaxis.setRange(ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05)

        self.xaxis.setRange(xmin, xmax)
        self.yaxis.setRange(ymin, ymax)

        poly = array_to_polygonf(points)
        self.series.replace(poly)

        self.line.replace([poly.first(), poly.last()])

        self.xaxis.applyNiceNumbers()
        self.yaxis.applyNiceNumbers()

    def setLine(self, x0: float, x1: float, gradient: float, intercept: float) -> None:
        self.line.replace(
            [
                QtCore.QPointF(x0, gradient * x0 + intercept),
                QtCore.QPointF(x1, gradient * x1 + intercept),
            ]
        )

    def showPointPosition(self, point: QtCore.QPointF, state: bool):
        self.label_series.setVisible(state)
        self.label_series.setPointLabelsVisible(state)
        if state:
            self.label_series.replace(0, point.x(), point.y())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.RightButton:
            self.chart().zoomReset()
        else:
            super().mouseReleaseEvent(event)
        self.xaxis.applyNiceNumbers()
        self.yaxis.applyNiceNumbers()
