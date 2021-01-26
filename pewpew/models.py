from PySide2 import QtCore

import numpy as np

from pewlib.calibration import Calibration

from pewpew.lib.numpyqt import NumpyArrayTableModel

from typing import Any, Tuple


class CalibrationPointsTableModel(NumpyArrayTableModel):
    def __init__(
        self,
        calibration: Calibration,
        axes: Tuple[int, int] = (0, 1),
        counts_editable: bool = False,
        parent: QtCore.QObject = None,
    ):
        self.calibration = calibration
        if self.calibration.points.size == 0:
            array = np.array([[np.nan, np.nan]], dtype=np.float64)
        else:
            array = self.calibration.points

        super().__init__(array, axes=axes, fill_value=np.nan, parent=parent)
        self.counts_editable = counts_editable

        self.dataChanged.connect(self.updateCalibration)
        self.rowsInserted.connect(self.updateCalibration)
        self.rowsRemoved.connect(self.updateCalibration)
        self.columnsInserted.connect(self.updateCalibration)
        self.columnsRemoved.connect(self.updateCalibration)
        self.modelReset.connect(self.updateCalibration)

    def setCalibration(self, calibration: Calibration, resize: bool = True) -> None:
        self.calibration = calibration

        if calibration.points.shape[1] != 2:  # pragma: no cover
            raise ValueError("Calibration points must have shape (n, 2).")

        if resize:
            self.blockSignals(True)
            self.setColumnCount(calibration.points.shape[self.axes[1]])
            self.setRowCount(calibration.points.shape[self.axes[0]])
            self.blockSignals(False)

        self.beginResetModel()

        new_array = np.full(self.array.shape, np.nan)
        if self.calibration.points.size > 0:
            min_row = np.min((new_array.shape[0], calibration.points.shape[0]))
            new_array[:min_row, :2] = self.calibration.points[:min_row]

        self.array = new_array

        self.endResetModel()

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> str:
        value = super().data(index, role)
        if value == "nan":
            return ""
        return value

    def setData(
        self, index: QtCore.QModelIndex, value: Any, role: int = QtCore.Qt.EditRole
    ) -> bool:
        return super().setData(index, np.nan if value == "" else value, role)

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():  # pragma: no cover
            return QtCore.Qt.ItemIsEnabled

        pos = (index.row(), index.column())

        flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if self.counts_editable or pos[self.axes[1]] == 0:
            flags = flags | QtCore.Qt.ItemIsEditable

        return flags

    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int
    ) -> str:
        if role != QtCore.Qt.DisplayRole:  # pragma: no cover
            return None

        labels = [("Concentration", "Counts"), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

        if orientation == QtCore.Qt.Horizontal:
            return labels[self.axes[0]][section]
        else:
            return labels[self.axes[1]][section]

    def updateCalibration(self, *args) -> None:
        if self.array.size == 0:
            self.calibration._points = np.empty((0, 2), dtype=np.float64)
        else:
            self.calibration.points = self.array[:, :2]

        print(self.array)
        self.calibration.update_linreg()
