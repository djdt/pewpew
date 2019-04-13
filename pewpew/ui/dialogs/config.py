from PyQt5 import QtWidgets
import copy

from pewpew.ui.dialogs.applydialog import ApplyDialog
from pewpew.ui.validators import DecimalValidator

from pewpew.lib.laser import LaserConfig
from pewpew.lib.krisskross import KrissKrossConfig


class ConfigDialog(ApplyDialog):
    def __init__(self, config: LaserConfig, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.config = copy.copy(config)

        # Line edits
        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setPlaceholderText(str(self.config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidator(0, 1e3, 0))
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setPlaceholderText(str(self.config.speed))
        self.lineedit_speed.setValidator(DecimalValidator(0, 1e3, 0))
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setPlaceholderText(str(self.config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidator(0, 1e3, 4))

        if isinstance(config, KrissKrossConfig):
            self.lineedit_warmup = QtWidgets.QLineEdit()
            self.lineedit_warmup.setPlaceholderText(str(self.config.warmup))
            self.lineedit_warmup.setValidator(DecimalValidator(0, 100, 1))
            self.spinbox_offsets = QtWidgets.QSpinBox()
            self.spinbox_offsets.setRange(2, 10)
            self.spinbox_offsets.setValue(
                self.config.subpixel_gcd.limit_denominator().denominator
            )
            self.check_horz = QtWidgets.QCheckBox("Horizontal layer first.")
            self.check_horz.setChecked(self.config.horizontal_first)

        # Form layout for line edits
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Spotsize (μm):", self.lineedit_spotsize)
        form_layout.addRow("Speed (μm):", self.lineedit_speed)
        form_layout.addRow("Scantime (s):", self.lineedit_scantime)

        if isinstance(config, KrissKrossConfig):
            form_layout.addRow("Warmup (s):", self.lineedit_warmup)
            form_layout.addRow("Subpixel width:", self.spinbox_offsets)
            form_layout.addRow(self.check_horz)

        # Checkbox
        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.check_all)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def updateConfig(self) -> None:
        if self.lineedit_spotsize.text() != "":
            self.config.spotsize = float(self.lineedit_spotsize.text())
        if self.lineedit_speed.text() != "":
            self.config.speed = float(self.lineedit_speed.text())
        if self.lineedit_scantime.text() != "":
            self.config.scantime = float(self.lineedit_scantime.text())
        if isinstance(self.config, KrissKrossConfig):
            if self.lineedit_warmup.text() != "":
                self.config.warmup = float(self.lineedit_warmup.text())
            self.config.set_subpixel_offsets(self.spinbox_offsets.value())
            self.config.horizontal_first = self.check_horz.isChecked()

    def apply(self) -> None:
        self.updateConfig()

    def accept(self) -> None:
        self.updateConfig()
        super().accept()
