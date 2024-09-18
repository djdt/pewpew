import numpy as np
from pewlib.calibration import Calibration
from PySide6 import QtCore

from pewpew.models import CalibrationPointsTableModel


def test_calibration_points_table_model():
    calibration = Calibration.from_points(np.array([[1.0, 1.0], [2.0, 2.0]]))
    model = CalibrationPointsTableModel(calibration, counts_editable=True)

    # Check starting shape
    assert model.columnCount() == 3
    assert model.rowCount() == 2
    assert np.all(model.array["x"] == [1.0, 2.0])
    assert np.all(model.array["y"] == [1.0, 2.0])
    assert np.all(model.array["weights"] == [1.0, 1.0])

    model.insertRow(1)
    assert model.rowCount() == 3
    assert calibration.points.shape == (3, 2)

    # Add data
    model.setData(model.index(0, 0), "0.0")
    model.setData(model.index(0, 1), "1.0")
    model.setData(model.index(1, 0), "1.0")
    model.setData(model.index(1, 1), "2.0")
    model.setData(model.index(2, 0), "2.0")
    model.setData(model.index(2, 1), "3.0")
    assert model.data(model.index(0, 1)) == "1.0"

    assert calibration.points.shape == (3, 2)
    assert np.all(calibration.points == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert np.isclose(calibration.intercept, 1.0)
    assert np.isclose(calibration.gradient, 1.0)

    model.setData(model.index(0, 1), "nan")
    assert calibration.points.shape == (3, 2)
    assert np.isnan(model.array["y"][0])
    assert model.data(model.index(0, 1)) == ""

    # Change calibration
    model.setCalibration(Calibration.from_points(np.array([[0.0, 0.0], [2.0, 4.0]])))

    assert model.columnCount() == 3
    assert model.rowCount() == 2

    # Check header
    assert model.headerData(0, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "x"
    assert model.headerData(1, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "y"
    assert model.headerData(0, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "A"

    assert model.flags(model.index(0, 0)) & QtCore.Qt.ItemIsEditable
    assert model.flags(model.index(0, 1)) & QtCore.Qt.ItemIsEditable


def test_calibration_points_table_model_flipped():
    # Starts with empty calibration
    calibration = Calibration()
    model = CalibrationPointsTableModel(
        calibration, orientation=QtCore.Qt.Orientation.Horizontal, counts_editable=False
    )

    # Check starting shape
    assert model.columnCount() == 1
    assert model.rowCount() == 3
    assert np.all(np.isnan(model.array["x"]))
    assert np.all(np.isnan(model.array["y"]))
    assert np.all(np.isnan(model.array["weights"]))

    assert model.insertColumns(1, 2)
    assert calibration.points.shape == (3, 2)

    # Add data
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

    # Change calibration
    model.setCalibration(Calibration.from_points(np.array([[0.0, 0.0], [2.0, 4.0]])))

    assert model.columnCount() == 2
    assert model.rowCount() == 3

    # Check header
    assert model.headerData(0, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "x"
    assert model.headerData(1, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "y"
    assert model.headerData(0, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "A"

    assert model.flags(model.index(0, 0)) & QtCore.Qt.ItemIsEditable
    assert not model.flags(model.index(1, 0)) & QtCore.Qt.ItemIsEditable


def test_calibration_points_table_model_weights():
    calibration = Calibration.from_points(
        np.array([[1.0, 1.0], [2.0, 4.0], [3.0, 5.0]]),
        weights=("test", np.array([0.4, 0.4, 0.2])),
    )
    model = CalibrationPointsTableModel(calibration, counts_editable=False)

    assert np.all(model.array["x"] == [1.0, 2.0, 3.0])
    assert np.all(model.array["y"] == [1.0, 4.0, 5.0])
    assert np.all(model.array["weights"] == [0.4, 0.4, 0.2])
    assert model.flags(model.index(0, 2)) & QtCore.Qt.ItemIsEditable
    model.setData(model.index(0, 2), "0.6")
    assert calibration._weights[0] == 0.6

    model.setWeighting("x")

    assert np.all(model.array["weights"] == [1.0, 2.0, 3.0])
    assert not model.flags(model.index(0, 2)) & QtCore.Qt.ItemIsEditable

    model.setWeighting("test")
    assert model.flags(model.index(0, 2)) & QtCore.Qt.ItemIsEditable
