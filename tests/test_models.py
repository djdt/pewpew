import numpy as np

from PySide2 import QtCore

from pewlib.calibration import Calibration

from pewpew.models import CalibrationPointsTableModel


def test_calibration_points_table_model():
    calibration = Calibration.from_points(np.array([[1.0, 1.0], [2.0, 2.0]]))
    model = CalibrationPointsTableModel(calibration, axes=(0, 1), counts_editable=True)

    # Check starting shape
    assert model.columnCount() == 3
    assert model.rowCount() == 2
    assert np.all(model.array == [[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]])

    model.setRowCount(3)
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
    assert np.isnan(model.array[0, 1])
    assert model.data(model.index(0, 1)) == ""

    # Change calibration
    model.setCalibration(Calibration.from_points(np.array([[0.0, 0.0], [2.0, 4.0]])))

    assert model.columnCount() == 3
    assert model.rowCount() == 2

    # Check header
    assert (
        model.headerData(0, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
        == "Concentration"
    )
    assert (
        model.headerData(1, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "Response"
    )
    assert model.headerData(0, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "A"

    assert model.flags(model.index(0, 0)) & QtCore.Qt.ItemIsEditable
    assert model.flags(model.index(0, 1)) & QtCore.Qt.ItemIsEditable


def test_calibration_points_table_model_flipped():
    # Starts with empty calibration
    calibration = Calibration()
    model = CalibrationPointsTableModel(calibration, axes=(1, 0), counts_editable=False)

    # Check starting shape
    assert model.columnCount() == 1
    assert model.rowCount() == 3
    assert np.all(np.isnan(model.array))

    model.setColumnCount(3)
    assert model.columnCount() == 3
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
    assert (
        model.headerData(0, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole)
        == "Concentration"
    )
    assert model.headerData(1, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "Response"
    assert model.headerData(0, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "A"

    assert model.flags(model.index(0, 0)) & QtCore.Qt.ItemIsEditable
    assert not model.flags(model.index(1, 0)) & QtCore.Qt.ItemIsEditable


def test_calibration_points_table_model_weights():
    calibration = Calibration.from_points(
        np.array([[1.0, 1.0], [2.0, 4.0], [3.0, 5.0]]),
        weights=("test", np.array([0.4, 0.4, 0.2])),
    )
    model = CalibrationPointsTableModel(calibration, axes=(0, 1), counts_editable=False)

    assert np.all(model.array == [[1.0, 1.0, 0.4], [2.0, 4.0, 0.4], [3.0, 5.0, 0.2]])
    assert model.flags(model.index(0, 2)) & QtCore.Qt.ItemIsEditable
    model.setData(model.index(0, 2), "0.6")
    assert calibration._weights[0] == 0.6

    model.setWeighting("x")

    assert np.all(model.array == [[1.0, 1.0, 1.0], [2.0, 4.0, 2.0], [3.0, 5.0, 3.0]])
    assert not model.flags(model.index(0, 2)) & QtCore.Qt.ItemIsEditable

    model.setWeighting("test")
    assert model.flags(model.index(0, 2)) & QtCore.Qt.ItemIsEditable
