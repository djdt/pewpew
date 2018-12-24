import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from util.calc import weighted_linreg
from gui.validators import DoublePrecisionDelegate

from typing import List, Tuple


class CopyableTable(QtWidgets.QTableWidget):
    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
            self._advance()
        elif event.key() in [QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete]:
            self._delete()
        elif event.matches(QtGui.QKeySequence.Copy):
            self._copy()
        elif event.matches(QtGui.QKeySequence.Cut):
            self._cut()
        elif event.matches(QtGui.QKeySequence.Paste):
            self._paste()
        else:
            super().keyPressEvent(event)

    def _advance(self) -> None:
        row = self.currentRow()
        if row + 1 < self.rowCount():
            self.setCurrentCell(row + 1, self.currentColumn())

    def _copy(self) -> None:
        selection = sorted(self.selectedIndexes(), key=lambda i: (i.row(), i.column()))
        data = (
            '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
            "<table><tr>"
        )
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

    def _cut(self) -> None:
        self._copy()
        self._delete()

    def _delete(self) -> None:
        for i in self.selectedItems():
            i.setText("")

    def _paste(self) -> None:
        text = QtWidgets.QApplication.clipboard().text("plain")
        selection = self.selectedIndexes()
        start_row = min(selection, key=lambda i: i.row()).row()
        start_column = min(selection, key=lambda i: i.column()).column()

        for row, row_text in enumerate(text[0].split("\n")):
            for column, text in enumerate(row_text.split("\t")):
                item = self.item(start_row + row, start_column + column)
                if item is not None and QtCore.Qt.ItemIsEditable | item.flags():
                    item.setText(text)


class CalibrationTable(CopyableTable):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Concentration", "Counts"])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.setItemDelegate(DoublePrecisionDelegate(4))

    def complete(self) -> bool:
        for row in range(0, self.rowCount()):
            for column in range(0, self.columnCount()):
                if self.item(row, column).text() == "":
                    return False
        return True

    def setRowCount(self, rows: int) -> None:
        current_rows = self.rowCount()
        super().setRowCount(rows)

        if current_rows < rows:
            self.setVerticalHeaderLabels(CalibrationTable.ROW_LABELS)
            for row in range(current_rows, rows):
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.setItem(row, 0, item)
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                # Non editable item
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(row, 1, item)

    def concentrations(self) -> List[str]:
        return [self.item(row, 0).text() for row in range(0, self.rowCount())]

    def counts(self) -> List[str]:
        return [self.item(row, 1).text() for row in range(0, self.rowCount())]

    def updateConcentrations(self, data: List[str] = None) -> None:
        for row in range(0, self.rowCount()):
            self.item(row, 0).setText(data[row] if data is not None else "")

    def updateCounts(self, data: np.ndarray) -> None:
        # Default one empty array
        if len(data) == 1:
            return
        sections = np.array_split(data, self.rowCount(), axis=0)

        for row in range(0, self.rowCount()):
            mean_conc = np.mean(sections[row])
            self.item(row, 1).setText(f"{mean_conc:.4f}")

    def calibrationResults(self, weighting: str = "x") -> Tuple[float, float, float]:
        """Returns tuple of the gradient intercept and r^2 for the current data.
        Does not check if the table is complete and can be parsed."""
        x = np.array(self.concentrations(), dtype=np.float64)
        y = np.array(self.counts(), dtype=np.float64)

        if weighting == "1/x":
            weights = 1.0 / x
        elif weighting == "1/(x^2)":
            weights = 1.0 / (x ** 2)
        else:  # Default is no weighting
            weights = None

        return weighted_linreg(x, y, w=weights)


class DetailedError(QtWidgets.QMessageBox):
    def __init__(
        self,
        level: QtWidgets.QMessageBox.Icon,
        title: str = "Error",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(level, title, message, QtWidgets.QMessageBox.NoButton, parent)

        self.setDetailedText(detailed_message)
        textedit = self.findChildren(QtWidgets.QTextEdit)
        if textedit[0] is not None:
            textedit[0].setFixedSize(640, 320)

    @staticmethod
    def info(
        title: str = "Error",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        return DetailedError(
            QtWidgets.QMessageBox.Information, title, message, detailed_message, parent
        ).exec()

    @staticmethod
    def warning(
        title: str = "Error",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        return DetailedError(
            QtWidgets.QMessageBox.Warning, title, message, detailed_message, parent
        ).exec()

    @staticmethod
    def critical(
        title: str = "Error",
        message: str = "",
        detailed_message: str = "",
        parent: QtWidgets.QWidget = None,
    ) -> QtWidgets.QMessageBox.StandardButton:
        return DetailedError(
            QtWidgets.QMessageBox.Critical, title, message, detailed_message, parent
        ).exec()


class MultipleDirDialog(QtWidgets.QFileDialog):
    def __init__(self, title: str, directory: str, parent: QtWidgets.QWidget = None):
        super().__init__(parent, title, directory)
        self.setFileMode(QtWidgets.QFileDialog.Directory)
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        for view in self.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
            if isinstance(view.model(), QtWidgets.QFileSystemModel):
                view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    @staticmethod
    def getExistingDirectories(
        title: str, directory: str, parent: QtWidgets.QWidget = None
    ) -> List[str]:
        dlg = MultipleDirDialog(title, directory, parent)
        if dlg.exec():
            return list(dlg.selectedFiles())
        else:
            return []
