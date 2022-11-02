from PySide6 import QtCore

import numpy as np

from pewlib.calibration import Calibration

from pewpew.lib.numpyqt import NumpyArrayTableModel

from typing import Any,  Tuple


class CalibrationPointsTableModel(NumpyArrayTableModel):
    """Model for calibration points and weights.

    To allow edit of points y (counts / response) set `counts_editable`.
    Weights are editable if not in 'Calibration.KNOWN_WEIGHTING'.

    Args:
        calibration
        axes: (0, 1) for points in columns, (1, 0) for points in rows
        counts_editable: allow edit of points y
        parent: parent object
    """

    def __init__(
        self,
        calibration: Calibration,
        axes: Tuple[int, int] = (0, 1),
        counts_editable: bool = False,
        parent: QtCore.QObject | None = None,
    ):
        self.calibration = calibration
        if self.calibration.points.size == 0:
            array = np.full((1, 3), np.nan)
        else:
            array = np.concatenate(
                (self.calibration.points, self.calibration.weights[:, None]), axis=1
            )

        super().__init__(array, axes=axes, fill_value=np.nan, parent=parent)
        self.counts_editable = counts_editable

        self.dataChanged.connect(self.updateCalibration)
        self.rowsInserted.connect(self.updateCalibration)
        self.rowsRemoved.connect(self.updateCalibration)
        self.columnsInserted.connect(self.updateCalibration)
        self.columnsRemoved.connect(self.updateCalibration)
        self.modelReset.connect(self.updateCalibration)

    def setCalibration(self, calibration: Calibration, resize: bool = True) -> None:
        """Update the model with a new calibration.

        If not `resize` then the points will be padded or trimmed.

        Args:
            calibration
            resize: resize the model
        """
        self.calibration = calibration

        if calibration.points.shape[1] != 2:  # pragma: no cover
            raise ValueError("Calibration points must have shape (n, 2).")

        if resize:
            newshape = (calibration.points.shape[0], 3)
            self.blockSignals(True)
            self.setColumnCount(newshape[self.axes[1]])
            self.setRowCount(newshape[self.axes[0]])
            self.blockSignals(False)

        self.beginResetModel()

        new_array = np.full(self.array.shape, np.nan)
        if self.calibration.points.size > 0:
            min_row = np.min((new_array.shape[0], calibration.points.shape[0]))
            new_array[:min_row, :2] = self.calibration.points[:min_row]
            new_array[:min_row, 2] = self.calibration.weights[:min_row]

        self.array = new_array

        self.endResetModel()

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> [str]:
        """Map np.nan to 'nan'."""
        value = super().data(index, role)
        if value == "nan":
            return ""
        return value

    def setData(
        self, index: QtCore.QModelIndex, value: Any, role: int = QtCore.Qt.EditRole
    ) -> bool:
        """Map 'nan' to np.nan."""
        return super().setData(index, np.nan if value == "" else value, role)

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():  # pragma: no cover
            return QtCore.Qt.ItemIsEnabled

        pos = (index.row(), index.column())

        weights_editable = self.calibration.weighting not in Calibration.KNOWN_WEIGHTING

        flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if pos[self.axes[1]] == 0:  # Concentrations row / column
            flags = flags | QtCore.Qt.ItemIsEditable
        elif pos[self.axes[1]] == 1 and self.counts_editable:  # Repsonses
            flags = flags | QtCore.Qt.ItemIsEditable
        elif pos[self.axes[1]] == 2 and weights_editable:  # Weights
            flags = flags | QtCore.Qt.ItemIsEditable

        return flags

    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int
    ) -> [str]:
        if role != QtCore.Qt.DisplayRole:  # pragma: no cover
            return None

        labels = [
            ("Concentration", "Response", "Weights"),
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ]

        if orientation == QtCore.Qt.Horizontal:
            return labels[self.axes[0]][section]
        else:
            return labels[self.axes[1]][section]

    def updateCalibration(self) -> None:
        """Update the internal Calibration, called on model changes."""
        if self.array.size == 0:  # pragma: no cover
            self.calibration._points = np.empty((0, 2), dtype=np.float64)
            self.calibration._weights = np.empty(0, dtype=np.float64)
        else:
            self.calibration.points = self.array[:, :2]
            if self.calibration.weighting not in Calibration.KNOWN_WEIGHTING:
                self.calibration._weights = self.array[:, 2]

        self.calibration.update_linreg()

    def setWeighting(self, weighting: str) -> None:
        """Sets weighting and updates weights.

        If weighting in 'Calibration.KNOWN_WEIGHTING' then weights are calculated.
        """
        if weighting in Calibration.KNOWN_WEIGHTING:
            self.calibration.weights = weighting
        else:
            self.calibration.weights = (weighting, self.array[:, 2])

        self.beginResetModel()
        self.array[:, 2] = self.calibration.weights
        self.endResetModel()
