from PyQt5 import QtWidgets

from pewpew.ui.dialogs.applydialog import ApplyDialog
from pewpew.ui.validators import PercentOrDecimalValidator

from typing import Tuple, Union


class ColorRangeDialog(ApplyDialog):
    def __init__(
        self,
        current_range: Tuple[Union[int, str], Union[int, str]],
        parent: QtWidgets.QWidget = None,
    ):
        self.range = current_range
        super().__init__(parent)
        self.setWindowTitle("Colormap Range")

        self.lineedit_min = QtWidgets.QLineEdit()
        self.lineedit_min.setPlaceholderText(str(current_range[0]))
        self.lineedit_min.setToolTip("Percentile for minium colormap value.")
        self.lineedit_min.setValidator(
            PercentOrDecimalValidator(parent=self.lineedit_min)
        )
        self.lineedit_max = QtWidgets.QLineEdit()
        self.lineedit_max.setPlaceholderText(str(current_range[1]))
        self.lineedit_max.setValidator(
            PercentOrDecimalValidator(parent=self.lineedit_max)
        )
        self.lineedit_max.setToolTip("Percentile for maximum colormap value.")

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Minimum:", self.lineedit_min)
        form_layout.addRow("Maximum:", self.lineedit_max)

        main_layout = QtWidgets.QVBoxLayout()

        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def updateRange(self) -> None:
        min, max = self.lineedit_min.text(), self.lineedit_max.text()
        if min == "":
            min = self.range[0]
        elif "%" not in min:
            min = float(min)
        if max == "":
            max = self.range[1]
        elif "%" not in max:
            max = float(max)
        self.range = (min, max)

    def apply(self) -> None:
        self.updateRange()

    def accept(self) -> None:
        self.updateRange()
        super().accept()
