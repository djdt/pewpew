import numpy as np
from pytestqt.qtbot import QtBot

from PySide2 import QtCore, QtGui

from pewpew.lib.numpyqt import (
    array_to_image,
    array_to_polygonf,
    polygonf_to_array,
    NumpyArrayTableModel,
)


def test_array_to_image():
    # Float image
    x = np.linspace(0.0, 1.0, 100, endpoint=True).reshape(10, 10).astype(np.float64)
    i = array_to_image(x)
    i.setColorTable(np.arange(256))
    assert i.format() == QtGui.QImage.Format_Indexed8
    assert i.width() == 10
    assert i.height() == 10
    assert i.pixel(0, 0) == 0
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
    assert i.pixel(0, 0) == (255 << 24)
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


def test_numpy_array_table_model(qtbot: QtBot):
    model = NumpyArrayTableModel(np.random.random((5, 3)))

    assert model.columnCount() == 3
    assert model.rowCount() == 5

    with qtbot.waitSignal(model.rowsInserted):
        model.insertRows(1, 1)
    assert model.array.shape == (6, 3)
    assert np.all(model.array[1, :] == model.fill_value)

    with qtbot.waitSignal(model.rowsRemoved):
        model.removeRows(1, 1)
    assert model.array.shape == (5, 3)

    with qtbot.waitSignal(model.columnsInserted):
        model.insertColumns(0, 2)
    assert model.array.shape == (5, 5)
    assert np.all(model.array[:, 0] == model.fill_value)

    with qtbot.waitSignal(model.columnsRemoved):
        model.removeColumns(0, 2)
    assert model.array.shape == (5, 3)

    assert model.data(model.index(0, 0)) == str(model.array[0, 0])
    assert model.data(model.index(0, -1)) is None
    assert model.data(model.index(0, 4)) is None
    assert model.data(model.index(10, 0)) is None

    with qtbot.waitSignal(model.dataChanged):
        assert model.setData(model.index(0, 0), np.nan)

    assert not model.setData(model.index(0, -1), np.nan)
    assert not model.setData(model.index(0, 3), np.nan)
    assert not model.setData(model.index(0, 0), np.nan, QtCore.Qt.DisplayRole)

    assert not model.setData(model.index(0, 0), "false")

    assert model.data(model.index(0, 0)) == "nan"

    assert model.flags(model.index(0, 0)) & QtCore.Qt.ItemIsEditable != 0

    assert model.headerData(0, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "0"


def test_numpy_array_table_model_flipped():
    model = NumpyArrayTableModel(np.random.random((5, 3)), axes=(1, 0))

    assert model.columnCount() == 5
    assert model.rowCount() == 3

    model.setRowCount(10)
    model.setColumnCount(10)
    assert model.rowCount() == 10
    assert model.columnCount() == 10

    model.setRowCount(2)
    model.setColumnCount(2)
    assert model.rowCount() == 2
    assert model.columnCount() == 2


def test_numpy_array_table_model_empty():
    model = NumpyArrayTableModel(np.random.random(5))
    assert model.rowCount() == 1
