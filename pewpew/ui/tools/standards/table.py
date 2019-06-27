from PyQt5 import QtWidgets
import numpy as np

from pewpew.ui.validators import DoublePrecisionDelegate

from pewpew.ui.widgets.basictable import BasicTableView
from pewpew.ui.tools.standards.calibrationpointstablemodel import (
    CalibrationPointsTableModel,
)


class StandardsTable(BasicTableView):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]
    COLUMN_CONC = 0
    COLUMN_COUNT = 1

    def __init__(self, calibration, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        model = CalibrationPointsTableModel(calibration, self)
        self.setModel(model)
        # self.setHorizontalHeader(["Concentration", "Counts"])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.setItemDelegate(DoublePrecisionDelegate(4))

    def isComplete(self) -> bool:
        if np.nan in self.model().array[:, 1]:
            return False
        if np.count_nonzero(~np.isnan(self.model().array[:, 0])) < 2:
            return False
        return True

    def setCounts(self, counts: np.ndarray) -> None:
        for i in range(0, self.model().rowCount()):
            print(i, counts[i], self.model().rowCount())
            self.model().setData(self.model().index(i, 1), counts[i])

    def setRowCount(self, rows: int) -> None:
        current_rows = self.model().rowCount()
        if current_rows < rows:
            self.model().insertRows(current_rows, rows - current_rows)
        elif current_rows > rows:
            self.model().removeRows(rows, current_rows - rows)
