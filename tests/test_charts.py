import numpy as np
from pytestqt.qtbot import QtBot

from pewpew.charts.calibration import CalibrationView
from pewpew.charts.colocal import ColocalisationView
from pewpew.charts.histogram import HistogramView


def test_calibration_view(qtbot: QtBot):
    chart = CalibrationView()
    qtbot.addWidget(chart)
    qtbot.waitExposed(chart)

    points = np.stack((np.arange(5.0), np.random.random(5)), axis=1)

    chart.drawPoints(points, name="test")
    chart.drawTrendline(np.ones_like(points))

    assert chart.readyForExport()
    data = chart.dataForExport()

    assert "concentration" in data
    assert np.all(data["concentration"] == points[:, 0])
    assert "response" in data
    assert np.all(data["response"] == points[:, 1])


def test_colocal_view(qtbot: QtBot):
    chart = ColocalisationView()
    qtbot.addWidget(chart)
    qtbot.waitExposed(chart)

    x = np.random.random(10)
    y = np.random.random(10)

    chart.drawPoints(x, y, 0.2, 0.2)
    chart.drawLine(1.0, 0.5)
    chart.drawThresholds(0.2, 0.2)


def test_histogram_view(qtbot: QtBot):
    chart = HistogramView()
    qtbot.addWidget(chart)
    qtbot.waitExposed(chart)

    x = np.random.random(100)

    chart.setHistogram(x)
    assert chart.readyForExport()

    data = chart.dataForExport()

    assert "counts" in data
    assert "bin_edges" in data
