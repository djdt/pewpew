from PyQt5 import QtGui, QtWidgets

from gui.dialogs.applydialog import ApplyDialog
from gui.validators import DecimalValidator

from typing import Tuple


class TrimDialog(ApplyDialog):
    def __init__(
        self, trim: Tuple[float, float] = (0.0, 0.0), parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Trim")
        self.trim = trim

        self.lineedit_left = QtWidgets.QLineEdit()
        self.lineedit_left.setPlaceholderText(str(trim[0]))
        self.lineedit_right = QtWidgets.QLineEdit()
        self.lineedit_right.setPlaceholderText(str(trim[1]))

        self.combo_trim = QtWidgets.QComboBox()
        self.combo_trim.addItems(["rows", "s", "μm"])
        self.combo_trim.currentIndexChanged.connect(self.comboTrim)
        self.combo_trim.setCurrentIndex(1)

        layout_trim = QtWidgets.QHBoxLayout()
        layout_trim.addWidget(QtWidgets.QLabel("Left:"))
        layout_trim.addWidget(self.lineedit_left)
        layout_trim.addWidget(QtWidgets.QLabel("Right:"))
        layout_trim.addWidget(self.lineedit_right)
        layout_trim.addWidget(self.combo_trim)

        self.check_all = QtWidgets.QCheckBox("Apply trim to all images.")

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_trim)
        layout_main.addWidget(self.check_all)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

    def comboTrim(self) -> None:
        if self.combo_trim.currentText() == "rows":
            self.lineedit_left.setValidator(QtGui.QIntValidator(0, 1e9))
            self.lineedit_right.setValidator(QtGui.QIntValidator(0, 1e9))
        else:
            self.lineedit_left.setValidator(DecimalValidator(0, 1e9, 2))
            self.lineedit_right.setValidator(DecimalValidator(0, 1e9, 2))

    def updateTrim(self) -> None:
        trim_left = (
            float(self.lineedit_left.text())
            if self.lineedit_left.text() != ""
            else self.trim[0]
        )
        trim_right = (
            float(self.lineedit_right.text())
            if self.lineedit_right.text() != ""
            else self.trim[1]
        )
        self.trim = (trim_left, trim_right)

    def apply(self) -> None:
        self.updateTrim()

    def accept(self) -> None:
        self.updateTrim()
        super().accept()
