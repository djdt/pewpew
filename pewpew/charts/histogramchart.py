from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np

from typing import Union


class HistogramChart(QtCharts.QChartView):
    def __init__(self, title: str = None, parent: QtWidgets.QWidget = None):
        super().__init__(QtCharts.QChart(), parent)
        # self.setRubberBand(QtCharts.QChartView.RectangleRubberBand)
        self.setMinimumSize(QtCore.QSize(640, 320))
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        if title is not None:
            self.chart().setTitle(title)

        self.chart().legend().hide()

        self.xaxis = QtCharts.QValueAxis()
        self.xaxis.setLabelsVisible(False)
        self.xaxis.setGridLineVisible(False)
        self.yaxis = QtCharts.QValueAxis()
        self.yaxis.setLabelsVisible(False)
        self.yaxis.setGridLineVisible(False)

        self.chart().addAxis(self.xaxis, QtCore.Qt.AlignBottom)
        self.chart().addAxis(self.yaxis, QtCore.Qt.AlignLeft)

        self.series = QtCharts.QBarSeries()
        self.series.setBarWidth(1.0)
        # self.series.setColor(QtCore.Qt.black)

        self.chart().addSeries(self.series)
        self.series.attachAxis(self.xaxis)
        self.series.attachAxis(self.yaxis)

        # self.series.hovered.connect(self.showPointPosition)

    def setHistogram(self, data: np.ndarray, bins: Union[int, str] = "fd") -> None:
        print('a')
        barset = QtCharts.QBarSet("histogram")

        hist, _edges = np.histogram(data, bins=bins)
        barset.append(list(hist))

        self.series.clear()
        self.series.append(barset)

        self.xaxis.setRange(-0.5, hist.size - 0.5)
        self.yaxis.setRange(0, np.amax(hist))
        self.yaxis.applyNiceNumbers()

    # def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
    #     if event.button() == QtCore.Qt.RightButton:
    #         self.chart().zoomReset()
    #     else:
    #         super().mouseReleaseEvent(event)
    #     self.xaxis.applyNiceNumbers()
    #     self.yaxis.applyNiceNumbers()
