import numpy as np

from PySide2 import QtCore

from pytestqt.qtbot import QtBot

from pewpew.lib.numpyqt import NumpyArrayTableModel


def test_numpy_array_table_model(qtbot: QtBot):
    model = NumpyArrayTableModel(np.random.random([5, 3]))

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

    with qtbot.waitSignal(model.dataChanged):
        model.setData(model.index(0, 0), np.nan)

    assert model.data(model.index(0, 0)) == "nan"

    assert model.flags(model.index(0, 0)) & QtCore.Qt.ItemIsEditable != 0

    assert model.headerData(0, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole) == "0"
