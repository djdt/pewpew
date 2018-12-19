from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np

from util.calc import weighted_linreg


class CalibrationTable(QtWidgets.QTableWidget):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]

    def __init__(self, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Concentration", "Counts"])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+c"), self).activated.connect(
            self._copy
        )

    def complete(self):
        for row in range(0, self.rowCount()):
            for column in range(0, self.columnCount()):
                if self.item(row, column).text() == "":
                    return False
        return True

    def setRowCount(self, rows):
        current_rows = self.rowCount()
        super().setRowCount(rows)

        if current_rows < rows:
            self.setVerticalHeaderLabels(CalibrationTable.ROW_LABELS)
            for row in range(current_rows, rows):
                item = QtWidgets.QTableWidgetItem()
                self.setItem(row, 0, item)
                item = QtWidgets.QTableWidgetItem()
                # Non editable item
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(row, 1, item)

    def updateCounts(self, data):
        # Default one empty array
        if len(data) == 1:
            return
        sections = np.array_split(data, self.rowCount(), axis=0)

        for row in range(0, self.rowCount()):
            mean_conc = np.mean(sections[row])
            self.item(row, 1).setText(str(mean_conc))

    def concentrations(self):
        return np.array(
            [float(self.item(row, 0).text()) for row in range(0, self.rowCount())],
            dtype=np.float64,
        )

    def counts(self):
        return np.array(
            [float(self.item(row, 1).text()) for row in range(0, self.rowCount())],
            dtype=np.float64,
        )

    def calibrationResults(self, weighting="x"):
        """Returns tuple of the gradient intercept and r^2 for the current data.
        Does not check if the table is complete and can be parsed."""
        x = self.concentrations()
        y = self.counts()

        if weighting == "1/x":
            weights = 1.0 / np.array(x, dtype=np.float64)
        elif weighting == "1/(x^2)":
            weights = 1.0 / (np.array(x, dtype=np.float64) ** 2)
        else:  # Default is no weighting
            weights = None

        return weighted_linreg(x, y, w=weights)

    def _copy(self):
        selection = sorted(self.selectedIndexes(), key=lambda i: (i.row(), i.column()))
        data = ('<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
                '<table><tr>')
        text = ""

        prev = None
        for i in selection:
            if prev is not None and prev.row() != i.row():  # New row
                data += "</tr><tr>"
                text += "\n"
            value = "" if i.data() is None else i.data()
            data += f"<td>{value}</td>"
            if i.column() != 0:
                text += "\t"
            text += f"{value}"
            prev = i
        data += "</tr></table>"

        mime = QtCore.QMimeData()
        mime.setHtml(data)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def _paste(self):
        pass

        # text = ""
        # for i in range(0, len(selection)):
        #     display_text = selection[i].data(QtCore.Qt.DisplayRole)
        #     if display_text is not None:
        #         text += display_text
        #     if i > 0 or selection[i - 1].row() != selection[i].row():
        #         text += "\n"
        #     else:
        #         text += "\t"
        # QtWidgets.QApplication.clipboard().setText(text)
