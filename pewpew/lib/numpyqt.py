from PySide2 import QtCore
import numpy as np

from typing import Any


class NumpyArrayTableModel(QtCore.QAbstractTableModel):
    def __init__(
        self, array: np.ndarray, hide_nan: bool = True, parent: QtCore.QObject = None
    ):
        super().__init__(parent)
        self.array = array
        self.fill_value = 0.0

    # Rows and Columns
    def columnCount(self, parent: QtCore.QModelIndex = None) -> int:
        if self.array.ndim > 1:
            return self.array.shape[1]
        else:
            return 1

    def rowCount(self, parent: QtCore.QModelIndex = None) -> int:
        return self.array.shape[0]

    def insertRows(
        self, position: int, rows: int, parent: QtCore.QModelIndex = None
    ) -> bool:
        self.beginInsertRows(parent, position, position + rows - 1)
        self.array = np.insert(
            self.array,
            position,
            np.full((rows, 1), self.fill_value, dtype=self.array.dtype),
            axis=0,
        )
        self.endInsertRows()
        return True

    def insertColumns(
        self, position: int, columns: int, parent: QtCore.QModelIndex = None
    ) -> bool:
        self.beginInsertColumns(parent, position, position + columns - 1)
        self.array = np.insert(
            self.array,
            position,
            np.full((columns, 1), self.fill_value, dtype=self.array.dtype),
            axis=1,
        )
        self.endInsertColumns()
        return True

    def removeRows(
        self, position: int, rows: int, parent: QtCore.QModelIndex = None
    ) -> bool:
        self.beginRemoveRows(parent, position, position + rows - 1)
        self.array = np.delete(self.array, np.arange(position, position + rows), axis=0)
        self.endRemoveRows()
        return True

    def removeColumns(
        self, position: int, columns: int, parent: QtCore.QModelIndex = None
    ) -> bool:
        self.beginRemoveColumns(parent, position, position + columns - 1)
        self.array = np.delete(
            self.array, np.arange(position, position + columns), axis=1
        )
        self.endRemoveColumns()
        return True

    # Data
    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> str:
        if not index.isValid():
            return None

        if index.row() >= self.rowCount() or index.column() >= self.columnCount():
            return None

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            value = self.array[index.row(), index.column()]
            return str(value)
        else:
            return None

    def setData(
        self, index: QtCore.QModelIndex, value: Any, role: int = QtCore.Qt.EditRole
    ) -> bool:
        if not index.isValid():
            return False

        if index.row() >= self.rowCount() or index.column() >= self.columnCount():
            return False

        if role == QtCore.Qt.EditRole:
            try:
                self.array[index.row(), index.column()] = value
                self.dataChanged.emit(index, index, [role])
                return True
            except ValueError:
                return False
        return False

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled

        return QtCore.Qt.ItemIsEditable | super().flags(index)

    # Header
    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int
    ) -> str:
        if role != QtCore.Qt.DisplayRole:
            return None

        return str(section)
