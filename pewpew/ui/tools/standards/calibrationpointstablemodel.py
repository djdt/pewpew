from PyQt5 import QtCore
import numpy as np

from pewpew.lib.numpyqt import NumpyArrayTableModel

from laserlib.calibration import LaserCalibration


class CalibrationPointsTableModel(NumpyArrayTableModel):
    def __init__(self, calibration: LaserCalibration, parent: QtCore.QObject = None):
        self.calibration = calibration
        if self.calibration.points.size == 0:
            points = np.array([[np.nan, np.nan]], dtype=np.float64)
        else:
            points = self.calibration.points
        super().__init__(points, parent)

        self.alphabet_rows = True
        self.fill_value = np.nan

        self.dataChanged.connect(self.updateCalibration)
        self.rowsInserted.connect(self.updateCalibration)
        self.rowsRemoved.connect(self.updateCalibration)
        self.modelReset.connect(self.updateCalibration)

    def setCalibration(self, calibration: LaserCalibration) -> None:
        self.beginResetModel()
        self.calibration = calibration
        new_array = np.full_like(self.array, np.nan)
        if self.calibration.points.size > 0:
            min_row = np.min((new_array.shape[0], self.calibration.points.shape[0]))
            new_array[:min_row] = self.calibration.points[:min_row]
        self.array = new_array
        self.endResetModel()

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

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled

        flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if index.column() == 0:
            flags = QtCore.Qt.ItemIsEditable | flags
        return flags

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

    def updateCalibration(self, *args) -> None:
        if np.count_nonzero(~np.isnan(self.array[:, 0])) == 0:
            self.calibration.points = np.array([], dtype=np.float64)
        else:
            self.calibration.points = self.array

        self.calibration.update_linreg()
