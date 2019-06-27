from PyQt5 import QtCore
import numpy as np

from pewpew.lib.numpyqt import NumpyArrayTableModel


class CalibrationPointsTableModel(NumpyArrayTableModel):
    def __init__(self, points: np.ndarray, parent: QtCore.QObject = None):
        if points is None:
            points = np.full((1, 2), np.nan, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Invalid array for points.")
        super().__init__(points.astype(np.float64), parent)

        self.alphabet_rows = True
        self.fill_value = np.nan

    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole
    ) -> QtCore.QVariant:
        value = super().data(index, role)
        if value == "nan":
            return ""
        return value

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: QtCore.QVariant,
        role: int = QtCore.Qt.EditRole,
    ) -> bool:
        return super().setData(index, np.nan if value == "" else value, role)

    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int
    ) -> QtCore.QVariant:
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            return ("Concentration", "Counts")[section]
        else:
            if self.alphabet_rows:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[section]
            return str(section)

    def insertColumns(
        self,
        position: int,
        columns: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        return False

    def removeColumns(
        self,
        position: int,
        columns: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        return False
