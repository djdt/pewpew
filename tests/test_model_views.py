import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from pewpew.lib.numpyqt import NumpyArrayTableModel
from pewpew.widgets.modelviews import BasicTable, BasicTableView


def test_basic_table_view(qtbot: QtBot):
    view = BasicTableView()
    qtbot.addWidget(view)

    x = np.random.random((5, 5))
    model = NumpyArrayTableModel(x)
    view.setModel(model)

    view.setCurrentIndex(view.indexAt(QtCore.QPoint(0, 0)))
    assert view.currentIndex().row() == 0
    view._advance()
    assert view.currentIndex().row() == 1
    view.selectionModel().select(
        view.indexAt(QtCore.QPoint(0, 0)), QtCore.QItemSelectionModel.Select
    )
    view._copy()
    mime_data = QtWidgets.QApplication.clipboard().mimeData()
    assert mime_data.text() == str(x[0, 0]) + "\n" + str(x[1, 0])
    view._cut()  # Same as _copy, _delete
    view._delete()
    assert not np.isnan(x[0, 0])  # Can't delete out of a numpy array
    QtWidgets.QApplication.clipboard().setText("2.0")
    view._paste()
    assert x[0, 0] == 2.0

    view.contextMenuEvent(
        QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0), QtCore.QPoint(0, 0)
        )
    )


def test_basic_table(qtbot: QtBot):
    table = BasicTable()
    qtbot.addWidget(table)

    table.setRowCount(2)
    table.setColumnCount(2)

    table.setItem(0, 0, QtWidgets.QTableWidgetItem("0"))
    table.setItem(0, 1, QtWidgets.QTableWidgetItem("0"))
    table.setItem(1, 0, QtWidgets.QTableWidgetItem("0"))
    table.setItem(1, 1, QtWidgets.QTableWidgetItem("d"))

    table.setColumnText(0, ["a", "c"])
    table.setRowText(0, ["a", "b"])

    assert table.columnText(0) == ["a", "c"]
    assert table.rowText(0) == ["a", "b"]

    table.setCurrentCell(0, 0)
    assert table.currentRow() == 0
    table._advance()
    assert table.currentRow() == 1

    table.clearSelection()
    table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(0, 0, 0, 1), True)
    table._copy()
    mime_data = QtWidgets.QApplication.clipboard().mimeData()
    assert mime_data.text() == "a\tb"

    table.clearSelection()
    table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(1, 0, 1, 1), True)
    table._cut()  # Same as _copy, _delete
    mime_data = QtWidgets.QApplication.clipboard().mimeData()
    assert mime_data.text() == "c\td"
    assert table.item(1, 0).text() == ""
    assert table.item(1, 1).text() == ""

    table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(0, 0, 1, 1), True)
    table._delete()
    assert table.item(0, 0).text() == ""
    assert table.item(0, 1).text() == ""

    QtWidgets.QApplication.clipboard().setText("1\t2\n3\t4")
    table._paste()
    assert table.item(0, 0).text() == "1"
    assert table.item(0, 1).text() == "2"
    assert table.item(1, 0).text() == "3"
    assert table.item(1, 1).text() == "4"

    table.contextMenuEvent(
        QtGui.QContextMenuEvent(
            QtGui.QContextMenuEvent.Mouse, QtCore.QPoint(0, 0), QtCore.QPoint(0, 0)
        )
    )
