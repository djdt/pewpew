from PyQt5 import QtCore, QtGui, QtWidgets

from typing import List


class BasicTable(QtWidgets.QTableWidget):
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

    def columnText(self, column: int) -> List[str]:
        return [self.item(row, column).text() for row in range(0, self.rowCount())]

    def rowText(self, row: int) -> List[str]:
        return [
            self.item(row, column).text() for column in range(0, self.columnCount())
        ]

    def setColumnText(self, column: int, text: List[str] = None) -> None:
        if text is not None:
            assert(len(text) == self.rowCount())
        for row in range(0, self.rowCount()):
            self.item(row, column).setText(text[row] if text is not None else "")

    def setRowText(self, row: int, text: List[str] = None) -> None:
        if text is not None:
            assert(len(text) == self.columnCount())
        for column in range(0, self.columnCount()):
            self.item(row, column).setText(text[column] if text is not None else "")
