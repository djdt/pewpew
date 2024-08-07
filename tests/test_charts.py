import numpy as np
from PySide6 import QtCharts, QtCore, QtGui
from pytestqt.qtbot import QtBot

from pewpew.charts.base import BaseChart
from pewpew.charts.calibration import CalibrationChart
from pewpew.charts.colocal import ColocalisationChart
from pewpew.charts.colors import light_theme
from pewpew.charts.histogram import HistogramChart


def test_base_chart(qtbot: QtBot):
    chart = BaseChart(QtCharts.QChart(), light_theme)
    qtbot.addWidget(chart)

    axis = QtCharts.QValueAxis()
    chart.addAxis(axis, QtCore.Qt.AlignBottom)
    assert axis.gridLineColor() == chart.theme["grid"]
    assert axis.minorGridLineColor() == chart.theme["grid"]
    assert axis.labelsColor() == chart.theme["text"]
    assert axis.linePen().color() == chart.theme["axis"]
    assert axis.titleBrush().color() == chart.theme["title"]

    chart.contextMenuEvent(
        QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0), QtCore.QPoint(0, 0)
        )
    )
    chart.copyToClipboard()


def test_calibration_chart(qtbot: QtBot):
    chart = CalibrationChart("Calibration")
    qtbot.addWidget(chart)
    points = np.stack((np.arange(5), np.arange(1, 6)), axis=1).astype(np.float64)
    chart.setPoints(points)
    chart.setLine(0.0, 6.0, 1.0, 1.0)
    chart.setText("Text")

    chart.show()
    qtbot.waitExposed(chart)

    assert not chart.label_series.pointLabelsVisible()
    chart.showPointPosition(QtCore.QPointF(0.0, 1.0), True)
    assert chart.label_series.pointLabelsVisible()


def test_colocal_chart(qtbot: QtBot):
    chart = ColocalisationChart("Colocal")
    qtbot.addWidget(chart)

    chart.drawPoints(np.random.random(10), np.random.random(10))
    chart.drawLine(1.0, 0.0)
    chart.drawThresholds(0.25, 0.75)


def test_histogram_chart(qtbot: QtBot):
    chart = HistogramChart("Histogram")
    qtbot.addWidget(chart)

    data = np.random.random(100)
    chart.setHistogram(data, bins=20)
    assert chart.series.barSets()[0].count() == 20
    chart.setHistogram(data, bins=100, max_bins=50)
    assert chart.series.barSets()[0].count() == 50
    chart.setHistogram(data, bins=10, min_bins=50)
    assert chart.series.barSets()[0].count() == 50
