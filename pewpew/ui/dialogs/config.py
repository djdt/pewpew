from PyQt5 import QtWidgets

from pewpew.ui.dialogs.applydialog import ApplyDialog
from pewpew.ui.validators import DecimalValidator


class ConfigDialog(ApplyDialog):
    def __init__(self, config: dict, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.spotsize = config["spotsize"]
        self.speed = config["speed"]
        self.scantime = config["scantime"]

        # Line edits
        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setPlaceholderText(str(self.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidator(0, 1e3, 2))
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setPlaceholderText(str(self.speed))
        self.lineedit_speed.setValidator(DecimalValidator(0, 1e3, 2))
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setPlaceholderText(str(self.scantime))
        self.lineedit_scantime.setValidator(DecimalValidator(0, 1e3, 4))

        # Form layout for line edits
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Spotsize (μm):", self.lineedit_spotsize)
        form_layout.addRow("Speed (μm):", self.lineedit_speed)
        form_layout.addRow("Scantime (s):", self.lineedit_scantime)

        # Checkbox
        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.check_all)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def updateConfig(self) -> None:
        if self.lineedit_spotsize.text() != "":
            self.spotsize = float(self.lineedit_spotsize.text())
        if self.lineedit_speed.text() != "":
            self.speed = float(self.lineedit_speed.text())
        if self.lineedit_scantime.text() != "":
            self.scantime = float(self.lineedit_scantime.text())

    def apply(self) -> None:
        self.updateConfig()

    def accept(self) -> None:
        self.updateConfig()
        super().accept()
