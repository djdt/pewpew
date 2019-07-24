from PySide2 import QtGui, QtWidgets

from pewpew.ui.dialogs.applydialog import ApplyDialog

from typing import Tuple


class FilteringDialog(ApplyDialog):
    def __init__(
        self,
        current_window: Tuple[int, int],
        current_threshold: int,
        parent: QtWidgets.QWidget = None,
    ):
        self.window = current_window
        self.threshold = current_threshold
        super().__init__(parent)
        self.setWindowTitle("Filtering Properties")

        self.lineedit_x = QtWidgets.QLineEdit()
        self.lineedit_x.setPlaceholderText(str(self.window[0]))
        self.lineedit_x.setValidator(QtGui.QIntValidator(0, 999))
        self.lineedit_y = QtWidgets.QLineEdit()
        self.lineedit_y.setPlaceholderText(str(self.window[1]))
        self.lineedit_y.setValidator(QtGui.QIntValidator(0, 999))

        self.lineedit_threshold = QtWidgets.QLineEdit()
        self.lineedit_threshold.setPlaceholderText(str(self.threshold))
        self.lineedit_threshold.setValidator(QtGui.QIntValidator(0, 99))

        window_layout = QtWidgets.QHBoxLayout()
        window_layout.addWidget(self.lineedit_y)
        window_layout.addWidget(self.lineedit_x)

        self.layout_form.addRow("Window:", window_layout)
        self.layout_form.addRow("Threshold:", self.lineedit_threshold)

    def updateProps(self) -> None:
        x, y = self.lineedit_x.text(), self.lineedit_y.text()
        threshold = self.lineedit_threshold.text()
        if x == "":
            x = self.window[0]
        if y == "":
            y = self.window[1]
        self.window = (int(x), int(y))
        if threshold != "":
            self.threshold = int(threshold)

    def apply(self) -> None:
        self.updateProps()

    def accept(self) -> None:
        self.updateProps()
        super().accept()
