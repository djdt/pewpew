import numpy as np
from PySide6 import QtCore, QtGui
from pytestqt.modeltest import ModelTester

from pewpew.lib.numpyqt import (
    NumpyRecArrayTableModel,
    array_to_image,
    array_to_polygonf,
    polygonf_to_array,
)


def test_array_to_image():
    # Float image
    x = np.linspace(0.0, 1.0, 100, endpoint=True).reshape(10, 10).astype(np.float64)
    x[1, 0] = np.nan
    i = array_to_image(x)
    i.setColorTable(np.arange(256))
    assert i.format() == QtGui.QImage.Format_Indexed8
    assert i.width() == 10
    assert i.height() == 10
    assert i.pixel(0, 0) == 1
    assert i.pixel(0, 1) == 0
    assert i.pixel(9, 9) == 255

    # Uint8 image
    x = np.arange(100).reshape(10, 10).astype(np.uint8)
    i = array_to_image(x)
    i.setColorTable(np.arange(256))
    assert i.format() == QtGui.QImage.Format_Indexed8
    assert i.pixel(0, 0) == 0
    assert i.pixel(9, 9) == 99

    # Float RGB image
    x = np.linspace(0.0, 1.0, 100, endpoint=True).reshape(10, 10).astype(np.float64)
    x = np.stack((x, x, x), axis=2)
    i = array_to_image(x)
    assert i.format() == QtGui.QImage.Format_RGB32
    assert i.pixel(0, 0) == (255 << 24) + (1 << 16) + (1 << 8) + 1
    assert i.pixel(9, 9) == (255 << 24) + (255 << 16) + (255 << 8) + 255

    # RGB image
    x = np.arange(100).reshape(10, 10).astype(np.uint8)
    x = np.stack((x, x, x), axis=2)
    i = array_to_image(x)
    assert i.format() == QtGui.QImage.Format_RGB32
    assert i.pixel(0, 0) == (255 << 24)
    assert i.pixel(9, 9) == (255 << 24) + (99 << 16) + (99 << 8) + 99


def test_array_to_polygonf():
    x = np.stack((np.arange(10), np.arange(10)), axis=1)
    poly = array_to_polygonf(x)
    assert poly.size() == 10
    for i in range(poly.size()):
        assert poly[i].x() == i
        assert poly[i].y() == i


def test_polygonf_to_array():
    poly = QtGui.QPolygonF()
    for i in range(10):
        poly.append(QtCore.QPointF(i, i))
    x = polygonf_to_array(poly)

    assert np.all(x[:, 0] == np.arange(10))
    assert np.all(x[:, 1] == np.arange(10))


def test_numpy_recarray_table_model(qtmodeltester: ModelTester):
    array = np.empty(10, dtype=[("str", "U16"), ("int", int), ("float", float)])
    array["str"] = "A"
    array["int"] = np.arange(10)
    array["float"] = np.random.random(10)

    model = NumpyRecArrayTableModel(
        array,
        fill_values={"U": "0", "i": 0},
        name_formats={"int": "{:.1f}"},
        name_flags={"str": ~QtCore.Qt.ItemFlag.ItemIsEditable},
    )

    assert model.columnCount() == 3
    assert model.rowCount() == 10
    # Header
    assert model.headerData(0, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "str"
    assert model.headerData(1, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "int"
    assert model.headerData(2, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == "float"

    for i in range(10):
        assert model.headerData(i, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == str(i)

    # Col flags
    assert model.flags(model.index(1, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert not model.flags(model.index(1, 0)) & QtCore.Qt.ItemFlag.ItemIsEditable

    # Insert
    model.insertRows(1, 1)
    assert model.array.shape == (11,)
    assert model.array[1]["str"] == "0"
    assert model.array[1]["int"] == 0
    assert np.isnan(model.array[1]["float"])

    # Data
    assert model.data(model.index(1, 1)) == "0.0"
    assert model.data(model.index(1, 2)) == ""  # nan

    model.setData(model.index(1, 1), 10, QtCore.Qt.EditRole)
    assert model.array["int"][1] == 10

    qtmodeltester.check(model, force_py=True)


def test_numpy_recarray_table_model_horizontal(qtmodeltester: ModelTester):
    array = np.empty(10, dtype=[("str", "U16"), ("int", int), ("float", float)])
    array["str"] = "A"
    array["int"] = np.arange(10)
    array["float"] = np.random.random(10)

    model = NumpyRecArrayTableModel(
        array,
        orientation=QtCore.Qt.Orientation.Horizontal,
        fill_values={"U": "0", "i": 0},
        name_formats={"int": "{:.1f}"},
        name_flags={"str": ~QtCore.Qt.ItemFlag.ItemIsEditable},
    )

    assert model.columnCount() == 10
    assert model.rowCount() == 3
    # Header
    assert model.headerData(0, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "str"
    assert model.headerData(1, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "int"
    assert model.headerData(2, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "float"

    for i in range(10):
        assert model.headerData(i, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == str(
            i
        )

    # Col flags
    assert model.flags(model.index(1, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable
    assert not model.flags(model.index(0, 1)) & QtCore.Qt.ItemFlag.ItemIsEditable

    # Insert
    model.insertColumns(1, 1)
    assert model.array.shape == (11,)
    assert model.array[1]["str"] == "0"
    assert model.array[1]["int"] == 0
    assert np.isnan(model.array[1]["float"])

    # Data
    assert model.data(model.index(1, 1)) == "0.0"
    assert model.data(model.index(2, 1)) == ""  # nan

    model.setData(model.index(1, 1), 10, QtCore.Qt.EditRole)
    assert model.array["int"][1] == 10

    qtmodeltester.check(model, force_py=True)
