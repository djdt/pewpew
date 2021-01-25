import numpy as np
from pytestqt.qtbot import QtBot

from pewlib.calibration import Calibration

from pewpew.models import CalibrationPointsTableModel


def test_calibration_points_table_model(qtbot: QtBot):
    calibration = Calibration()
    model = CalibrationPointsTableModel(calibration, axes=(0, 1))
    qtbot.addWidget(model)

    assert model.columnCount() == 2
    assert model.rowCount() == 1
    assert np.all(np.isnan(model.array))

    model.setRowCount(3)
    assert model.rowCount() == 3
    assert calibration.points.shape == (0, 2)

    model.setData(model.index(0, 0), "0.0")
    model.setData(model.index(0, 1), "1.0")
    model.setData(model.index(1, 0), "1.0")
    model.setData(model.index(1, 1), "2.0")
    model.setData(model.index(2, 0), "2.0")
    model.setData(model.index(2, 1), "3.0")

    assert calibration.points.shape == (3, 2)
    assert np.all(calibration.points == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert np.isclose(calibration.intercept, 1.0)
    assert np.isclose(calibration.gradient, 1.0)

    model.setCalibration(Calibration.from_points([[0.0, 0.0], [2.0, 4.0]]))

    assert model.columnCount() == 2
    assert model.rowCount() == 2


def test_calibration_points_table_model_flipped(qtbot: QtBot):
    calibration = Calibration()
    model = CalibrationPointsTableModel(calibration, axes=(1, 0))
    qtbot.addWidget(model)

    assert model.columnCount() == 1
    assert model.rowCount() == 2
    assert np.all(np.isnan(model.array))

    model.setColumnCount(3)
    assert model.columnCount() == 3
    assert calibration.points.shape == (0, 2)

    model.setData(model.index(0, 0), "0.0")
    model.setData(model.index(1, 0), "1.0")
    model.setData(model.index(0, 1), "1.0")
    model.setData(model.index(1, 1), "2.0")
    model.setData(model.index(0, 2), "2.0")
    model.setData(model.index(1, 2), "3.0")

    assert calibration.points.shape == (3, 2)
    assert np.all(calibration.points == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert np.isclose(calibration.intercept, 1.0)
    assert np.isclose(calibration.gradient, 1.0)

    model.setCalibration(Calibration.from_points([[0.0, 0.0], [2.0, 4.0]]))

    assert model.columnCount() == 2
    assert model.rowCount() == 2
