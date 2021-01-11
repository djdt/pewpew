from PySide2 import QtCore, QtGui

import ctypes
import numpy as np
import shiboken2

from typing import Any


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
    def __init__(self, array: np.ndarray, parent: QtCore.QObject = None):
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
            axis=0,
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
            axis=1,
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
        self.array = np.delete(self.array, np.arange(position, position + rows), axis=0)
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
            self.array, np.arange(position, position + columns), axis=1
        )
        self.endRemoveColumns()
        return True

    # Data
    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> str:
        if not index.isValid():
            return None

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            value = self.array[index.row(), index.column()]
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
                self.array[index.row(), index.column()] = value
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
