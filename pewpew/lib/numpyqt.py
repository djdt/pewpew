from PySide2 import QtCore, QtGui

import ctypes
import numpy as np
import shiboken2

from typing import Any, Tuple


def array_to_image(array: np.ndarray) -> QtGui.QImage:
    array = np.atleast_2d(array)

    # Clip float arrays to 0.0 - 1.0 then convert to uint8
    if array.dtype in [np.float32, np.float64]:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).astype(np.uint8)

    # 3D arrays interpreted as RGB
    if array.ndim == 3:
        array = array.astype(np.uint32)
        array = (array[:, :, 0] << 16) + (array[:, :, 1] << 8) + array[:, :, 2]
        array += 255 << 24

    if array.dtype == np.uint8:
        image_format = QtGui.QImage.Format_Indexed8
    elif array.dtype == np.uint32:
        image_format = QtGui.QImage.Format_RGB32

    image = QtGui.QImage(
        array.data, array.shape[1], array.shape[0], array.strides[0], image_format
    )
    image._array = array
    return image


def polygonf_to_array(polygon: QtGui.QPolygonF) -> np.ndarray:
    buf = (ctypes.c_double * 2 * polygon.length()).from_address(
        shiboken2.getCppPointer(polygon.data())[0]
    )
    return np.frombuffer(buf, dtype=np.float64).reshape(-1, 2)


def array_to_polygonf(array: np.ndarray) -> QtGui.QPolygonF:
    assert array.ndim == 2
    assert array.shape[1] == 2

    polygon = QtGui.QPolygonF(array.shape[0])

    buf = (ctypes.c_double * array.size).from_address(
        shiboken2.getCppPointer(polygon.data())[0]
    )

    memory = np.frombuffer(buf, np.float64)
    memory[:] = array.ravel()
    return polygon


class NumpyArrayTableModel(QtCore.QAbstractTableModel):
    def __init__(
        self,
        array: np.ndarray,
        axes: Tuple[int, int] = (0, 1),
        fill_value: float = 0.0,
        parent: QtCore.QObject = None,
    ):
        assert array.ndim == 2

        super().__init__(parent)
        self.axes = axes
        self.array = array
        self.fill_value = fill_value

    # Rows and Columns
    def columnCount(self, parent: QtCore.QModelIndex = None) -> int:
        return self.array.shape[self.axes[1]]

    def rowCount(self, parent: QtCore.QModelIndex = None) -> int:
        return self.array.shape[self.axes[0]]

    def insertRows(
        self,
        position: int,
        rows: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        self.beginInsertRows(parent, position, position + rows - 1)
        self.array = np.insert(
            self.array,
            position,
            np.full((rows, 1), self.fill_value, dtype=self.array.dtype),
            axis=self.axes[0],
        )
        self.endInsertRows()
        return True

    def insertColumns(
        self,
        position: int,
        columns: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        self.beginInsertColumns(parent, position, position + columns - 1)
        self.array = np.insert(
            self.array,
            position,
            np.full((columns, 1), self.fill_value, dtype=self.array.dtype),
            axis=self.axes[1],
        )
        self.endInsertColumns()
        return True

    def removeRows(
        self,
        position: int,
        rows: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        self.beginRemoveRows(parent, position, position + rows - 1)
        self.array = np.delete(
            self.array, np.arange(position, position + rows), axis=self.axes[0]
        )
        self.endRemoveRows()
        return True

    def removeColumns(
        self,
        position: int,
        columns: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        self.beginRemoveColumns(parent, position, position + columns - 1)
        self.array = np.delete(
            self.array, np.arange(position, position + columns), axis=self.axes[1]
        )
        self.endRemoveColumns()
        return True

    def setColumnCount(self, columns: int) -> None:
        current_columns = self.columnCount()
        if current_columns < columns:
            self.insertColumns(current_columns, columns - current_columns)
        elif current_columns > columns:
            self.removeColumns(columns, current_columns - columns)

    def setRowCount(self, rows: int) -> None:
        current_rows = self.rowCount()
        if current_rows < rows:
            self.insertRows(current_rows, rows - current_rows)
        elif current_rows > rows:
            self.removeRows(rows, current_rows - rows)

    # Data
    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> str:
        if not index.isValid():
            return None

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            pos = [index.row(), index.column()]
            value = self.array[pos[self.axes[0]], pos[self.axes[1]]]
            return str(value)
        else:  # pragma: no cover
            return None

    def setData(
        self, index: QtCore.QModelIndex, value: Any, role: int = QtCore.Qt.EditRole
    ) -> bool:
        if not index.isValid():
            return False

        if role == QtCore.Qt.EditRole:
            try:
                pos = [index.row(), index.column()]
                self.array[pos[self.axes[0]], pos[self.axes[1]]] = value
                self.dataChanged.emit(index, index, [role])
                return True
            except ValueError:
                return False
        return False

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():  # pragma: no cover
            return QtCore.Qt.ItemIsEnabled

        return QtCore.Qt.ItemIsEditable | super().flags(index)

    # Header
    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: QtCore.Qt.ItemDataRole,
    ) -> str:
        if role != QtCore.Qt.DisplayRole:  # pragma: no cover
            return None

        return str(section)
