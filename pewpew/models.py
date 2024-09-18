from typing import Any

import numpy as np
from pewlib.calibration import Calibration
from PySide6 import QtCore

from pewpew.lib.numpyqt import NumpyRecArrayTableModel


class CalibrationPointsTableModel(NumpyRecArrayTableModel):
    """Model for calibration points and weights.

    To allow edit of points y (counts / response) set `counts_editable`.
    Weights are editable if not in 'Calibration.KNOWN_WEIGHTING'.

    Args:
        calibration
        counts_editable: allow edit of points y
        orientation: direction of array data, vertical = names in columns
        parent: parent object
    """

    def __init__(
        self,
        calibration: Calibration,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Vertical,
        counts_editable: bool = False,
        parent: QtCore.QObject | None = None,
    ):
        self.calibration = calibration
        if self.calibration.points.size == 0:
            array = np.full(
                1, np.nan, dtype=[("x", float), ("y", float), ("weights", float)]
            )
        else:
            array = np.empty(
                len(self.calibration.points),
                dtype=[("x", float), ("y", float), ("weights", float)],
            )
            array["x"] = self.calibration.x
            array["y"] = self.calibration.y
            array["weights"] = self.calibration.weights

        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        if counts_editable:
            flags = flags | QtCore.Qt.ItemFlag.ItemIsEditable
        super().__init__(
            array, orientation=orientation, name_flags={"y": flags}, parent=parent
        )

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
            shape = len(calibration.points)
        else:
            shape = len(self.array)

        self.beginResetModel()

        new_array = np.full(shape, np.nan, dtype=self.array.dtype)
        if self.calibration.points.size > 0:
            min_row = np.min((new_array.shape[0], calibration.points.shape[0]))
            new_array["x"][:min_row] = self.calibration.x[:min_row]
            new_array["y"][:min_row] = self.calibration.y[:min_row]
            new_array["weights"][:min_row] = self.calibration.weights[:min_row]

        self.array = new_array

        self.endResetModel()

    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole
    ) -> str | None:
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
            return 0

        flags = super().flags(index)

        idx = (
            index.column()
            if self.orientation == QtCore.Qt.Orientation.Vertical
            else index.row()
        )
        name = self.array.dtype.names[idx]

        weights_editable = self.calibration.weighting not in Calibration.KNOWN_WEIGHTING
        if not weights_editable and name == "weights":
            flags &= ~QtCore.Qt.ItemFlag.ItemIsEditable

        return flags

    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int
    ) -> str | None:
        if role != QtCore.Qt.DisplayRole:  # pragma: no cover
            return None

        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        if orientation == self.orientation:
            return labels[section]
        else:
            return self.array.dtype.names[section]

    def updateCalibration(self) -> None:
        """Update the internal Calibration, called on model changes."""
        if self.array.size == 0:  # pragma: no cover
            self.calibration._points = np.empty((0, 2), dtype=np.float64)
            self.calibration._weights = np.empty(0, dtype=np.float64)
        else:
            self.calibration.points = np.stack(
                (self.array["x"], self.array["y"]), axis=1
            )
            if self.calibration.weighting not in Calibration.KNOWN_WEIGHTING:
                self.calibration._weights = self.array["weights"]

        self.calibration.update_linreg()

    def setWeighting(self, weighting: str) -> None:
        """Sets weighting and updates weights.

        If weighting in 'Calibration.KNOWN_WEIGHTING' then weights are calculated.
        """
        if weighting in Calibration.KNOWN_WEIGHTING:
            self.calibration.weights = weighting
        else:
            self.calibration.weights = (weighting, self.array["weights"])

        self.beginResetModel()
        self.array["weights"] = self.calibration.weights
        self.endResetModel()
