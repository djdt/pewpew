import numpy as np
from PySide6 import QtCharts, QtCore, QtGui
from pytestqt.qtbot import QtBot

from pewpew.charts.calibration import CalibrationView
from pewpew.charts.colocal import ColocalisationView
from pewpew.charts.histogram import HistogramView


def test_calibration_view(qtbot: QtBot):
    chart = CalibrationView()
    qtbot.addWidget(chart)

    points = np.stack((np.arange(5.0), np.random.random(5)), axis=1)


def test_colocal_view(qtbot: QtBot):
    chart = ColocalisationView()
    qtbot.addWidget(chart)



def test_histogram_view(qtbot: QtBot):
    chart = HistogramView()
    qtbot.addWidget(chart)

