import ctypes
from typing import Any

import numpy as np
import shiboken6
from PySide6 import QtCore, QtGui


def array_to_image(array: np.ndarray) -> QtGui.QImage:
    """Converts a numpy array to a Qt image."""

    array = np.atleast_2d(array)

    # Clip float arrays to 0.0 - 1.0 then convert to uint8
    # Data is set to 1 - 255, NaNs are set to 0
    if array.dtype in [np.float32, np.float64]:
        nans = np.isnan(array)
        array = np.clip(array, 0.0, 1.0)
        with np.errstate(invalid="ignore"):
            array = (array * 254.0).astype(np.uint8) + 1
        array[nans] = 0

    # 3D arrays interpreted as RGB
    if array.ndim == 3:
        array = array.astype(np.uint32)
        array = (array[:, :, 0] << 16) + (array[:, :, 1] << 8) + array[:, :, 2]
        array += 255 << 24

    if array.dtype == np.uint8:
        image_format = QtGui.QImage.Format.Format_Indexed8
    elif array.dtype == np.uint32:
        image_format = QtGui.QImage.Format.Format_RGB32
    else:
        raise ValueError(f"Unknown image format for {array.dtype}.")

    array = np.ascontiguousarray(array)
    image = QtGui.QImage(
        array.data, array.shape[1], array.shape[0], array.strides[0], image_format
    )
    image._array = array
    return image


def image_to_array(image: QtGui.QImage, grey: bool = True) -> np.ndarray:
    if image.isGrayscale():
        image = image.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
        channels = 1
    else:
        image = image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        channels = 4

    array = np.array(image.constBits(), np.uint8).reshape(
        (image.height(), image.width(), channels)
    )

    return array


def array_to_polygonf(array: np.ndarray) -> QtGui.QPolygonF:
    """Converts a numpy array of shape (n, 2) to a Qt polygon."""
    assert array.ndim == 2
    assert array.shape[1] == 2

    polygon = QtGui.QPolygonF()
    polygon.resize(array.shape[0])

    buf = (ctypes.c_double * array.size).from_address(
        shiboken6.getCppPointer(polygon.data())[0]  # type: ignore
    )

    memory = np.frombuffer(buf, np.float64)
    memory[:] = array.ravel()
    return polygon


def polygonf_to_array(polygon: QtGui.QPolygonF) -> np.ndarray:
    """Converts a Qt polygon to a numpy array of shape (n, 2)."""
    buf = (ctypes.c_double * 2 * polygon.length()).from_address(
        shiboken6.getCppPointer(polygon.data())[0]  # type: ignore
    )
    return np.frombuffer(buf, dtype=np.float64).reshape(-1, 2)


class NumpyRecArrayTableModel(QtCore.QAbstractTableModel):
    """Access a numpy structured array through a table.

    Args:
        array: 1d array
        orientation: direction of array data, vertical = names in columns
        fill_values: default value for missing data for each type
        name_formats: dict of array names and formatting strings for display
        name_flags: dict of array names and model flags
        parent: parent object
    """

    def __init__(
        self,
        array: np.ndarray,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Vertical,
        fill_values: dict[str, Any] | None = None,
        name_formats: dict[str, str] | None = None,
        name_flags: dict[str, QtCore.Qt.ItemFlag] | None = None,
        parent: QtCore.QObject | None = None,
    ):
        assert array.ndim == 1
        assert array.dtype.names is not None

        super().__init__(parent)

        self.array = array
        self.orientation = orientation

        self.fill_values = {"f": np.nan, "U": "", "i": -1, "u": 0}
        if fill_values is not None:
            self.fill_values.update(fill_values)

        self.name_formats = {}
        if name_formats is not None:
            self.name_formats.update(name_formats)

        self.name_flags = {}
        if name_flags is not None:
            self.name_flags.update(name_flags)

    # Rows and Columns
    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            return self.array.shape[0]
        else:
            return len(self.array.dtype.names)  # type: ignore

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            return len(self.array.dtype.names)  # type: ignore
        else:
            return self.array.shape[0]

    # Data
    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if not index.isValid() or self.array.dtype.names is None:
            return None

        row, column = index.row(), index.column()
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            row, column = column, row

        if role in (
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ):
            name = self.array.dtype.names[column]
            value = self.array[name][row]
            if np.isreal(value) and np.isnan(value):
                return ""
            return self.name_formats.get(name, "{}").format(value)
        else:  # pragma: no cover
            return None

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> bool:
        if not index.isValid() or self.array.dtype.names is None:
            return False

        row, column = index.row(), index.column()
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            row, column = column, row

        name = self.array.dtype.names[column]
        if role == QtCore.Qt.ItemDataRole.EditRole:
            if value == "":
                value = self.fill_values[self.array[name].dtype.kind]

            try:
                self.array[name][row] = value
            except ValueError:
                self.array[name][row] = self.fill_values[self.array[name].dtype.kind]

            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlags:
        if not index.isValid() or self.array.dtype.names is None:  # pragma: no cover
            return 0

        idx = (
            index.column()
            if self.orientation == QtCore.Qt.Orientation.Vertical
            else index.row()
        )

        name = self.array.dtype.names[idx]
        return self.name_flags.get(
            name, super().flags(index) | QtCore.Qt.ItemFlag.ItemIsEditable
        )

    # Header
    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if role != QtCore.Qt.ItemDataRole.DisplayRole:  # pragma: no cover
            return None

        if orientation == self.orientation:
            return str(section)
        else:
            return self.array.dtype.names[section]

    def insertColumns(
        self,
        pos: int,
        count: int,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        if self.orientation == QtCore.Qt.Orientation.Vertical:
            raise NotImplementedError("name insert is not implemented")

        self.beginInsertColumns(parent, pos, pos + count - 1)
        empty = np.array(
            [
                tuple(
                    self.fill_values[d.kind]
                    for d, v in self.array.dtype.fields.values()
                )
            ],
            dtype=self.array.dtype,
        )
        self.array = np.insert(self.array, pos, np.full(count, empty))
        self.endInsertColumns()
        return True

    def insertRows(
        self,
        pos: int,
        count: int,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            raise NotImplementedError("name insert is not implemented")

        self.beginInsertRows(parent, pos, pos + count - 1)
        empty = np.array(
            [
                tuple(
                    self.fill_values[d.kind]
                    for d, v in self.array.dtype.fields.values()
                )
            ],
            dtype=self.array.dtype,
        )
        self.array = np.insert(self.array, pos, np.full(count, empty))
        self.endInsertRows()
        return True

    def removeColumns(
        self,
        pos: int,
        count: int,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        if self.orientation == QtCore.Qt.Orientation.Vertical:
            raise NotImplementedError("name insert is not implemented")

        self.beginRemoveColumns(parent, pos, pos + count - 1)
        self.array = np.delete(self.array, np.arange(pos, pos + count))
        self.endRemoveColumns()
        return True

    def removeRows(
        self,
        pos: int,
        count: int,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            raise NotImplementedError("name insert is not implemented")

        self.beginRemoveRows(parent, pos, pos + count - 1)
        self.array = np.delete(self.array, np.arange(pos, pos + count))
        self.endRemoveRows()
        return True
