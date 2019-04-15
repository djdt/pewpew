from PyQt5 import QtCore, QtWidgets

from pewpew.ui.dialogs.applydialog import ApplyDialog
from pewpew.ui.validators import DecimalValidator, DecimalValidatorNoZero

from pewpew.lib.laser import Laser

from typing import Dict, Tuple


class CalibrationDialog(ApplyDialog):
    def __init__(
        self, laser: Laser, current_isotope: str, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.laser = laser

        self.calibration: Dict[str, Tuple[float, float, str]] = {}
        for k, v in self.laser.data.items():
            self.calibration[k] = (v.gradient, v.intercept, v.unit)

        # LIne edits
        self.lineedit_gradient = QtWidgets.QLineEdit()
        self.lineedit_gradient.setValidator(DecimalValidatorNoZero(-1e10, 1e10, 4))
        self.lineedit_gradient.setPlaceholderText("1.0")
        self.lineedit_intercept = QtWidgets.QLineEdit()
        self.lineedit_intercept.setValidator(DecimalValidator(-1e10, 1e10, 4))
        self.lineedit_intercept.setPlaceholderText("0.0")
        self.lineedit_unit = QtWidgets.QLineEdit()
        self.lineedit_unit.setPlaceholderText("")

        # Form layout for line edits
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Gradient:", self.lineedit_gradient)
        form_layout.addRow("Intercept:", self.lineedit_intercept)
        form_layout.addRow("Unit:", self.lineedit_unit)

        # Isotope combo
        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.addItems(self.laser.isotopes())
        self.combo_isotopes.setCurrentText(current_isotope)
        self.previous_index = self.combo_isotopes.currentIndex()
        self.combo_isotopes.currentIndexChanged.connect(self.comboChanged)

        # Check all
        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")

        # Dialog buttons
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.combo_isotopes, 1, QtCore.Qt.AlignRight)
        main_layout.addWidget(self.check_all)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

        self.updateLineEdits()

    def updateLineEdits(self) -> None:
        name = self.combo_isotopes.currentText()

        gradient = self.calibration[name][0]
        if gradient == 1.0:
            self.lineedit_gradient.clear()
        else:
            self.lineedit_gradient.setText(str(gradient))
        intercept = self.calibration[name][1]
        if intercept == 0.0:
            self.lineedit_intercept.clear()
        else:
            self.lineedit_intercept.setText(str(intercept))
        unit = self.calibration[name][2]
        if unit is None:
            self.lineedit_unit.clear()
        else:
            self.lineedit_unit.setText(str(unit))

    def updateCalibration(self, name: str) -> None:
        gradient = self.lineedit_gradient.text()
        intercept = self.lineedit_intercept.text()
        unit = self.lineedit_unit.text()

        m, b, u = self.calibration[name]
        if gradient != "":
            m = float(gradient)
        if intercept != "":
            b = float(intercept)
        if unit != "":
            u = unit
        self.calibration[name] = (m, b, u)

    def comboChanged(self) -> None:
        previous = self.combo_isotopes.itemText(self.previous_index)
        self.updateCalibration(previous)
        self.updateLineEdits()
        self.previous_index = self.combo_isotopes.currentIndex()

    def apply(self) -> None:
        self.updateCalibration(self.combo_isotopes.currentText())

    def accept(self) -> None:
        self.updateCalibration(self.combo_isotopes.currentText())
        super().accept()
