from PyQt5 import QtCore, QtWidgets

from pewpew.ui.validators import DoublePrecisionDelegate

from .calibrationmodel import CalibrationModel


class StandardsTable(QtWidgets.QTableView):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]
    COLUMN_CONC = 0
    COLUMN_COUNT = 1

    def __init__(self, calibration, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setModel(CalibrationModel(calibration, self))
        # self.setHorizontalHeader(["Concentration", "Counts"])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.setItemDelegate(DoublePrecisionDelegate(4))

    def isComplete(self) -> bool:
        num_points = 0
        for row in range(0, self.rowCount()):
            if self.item(row, StandardsTable.COLUMN_COUNT).text() == "":
                return False
            if self.item(row, StandardsTable.COLUMN_CONC).text() != "":
                num_points += 1
        return num_points > 1

    def setRowCount(self, rows: int) -> None:
        current_rows = self.rowCount()
        super().setRowCount(rows)

        if current_rows < rows:
            self.setVerticalHeaderLabels(StandardsTable.ROW_LABELS)
            for row in range(current_rows, rows):
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.setItem(row, 0, item)
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                # Non editable item
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(row, 1, item)
