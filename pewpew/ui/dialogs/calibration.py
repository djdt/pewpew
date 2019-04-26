from PyQt5 import QtCore, QtWidgets
import copy

from pewpew.ui.dialogs.applydialog import ApplyDialog
from pewpew.ui.validators import DecimalValidator, DecimalValidatorNoZero

from laserlib.calibration import LaserCalibration

from typing import Dict


class CalibrationDialog(ApplyDialog):
    def __init__(
        self,
        calibrations: Dict[str, LaserCalibration],
        current_isotope: str,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.calibrations = copy.deepcopy(calibrations)

        # LIne edits
        self.lineedit_gradient = QtWidgets.QLineEdit()
        self.lineedit_gradient.setValidator(DecimalValidatorNoZero(-1e10, 1e10, 4))
        self.lineedit_gradient.setPlaceholderText("1.0")
        self.lineedit_intercept = QtWidgets.QLineEdit()
        self.lineedit_intercept.setValidator(DecimalValidator(-1e10, 1e10, 4))
        self.lineedit_intercept.setPlaceholderText("0.0")
        self.lineedit_unit = QtWidgets.QLineEdit()
        self.lineedit_unit.setPlaceholderText("")

        # Isotope combo
        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.addItems(self.calibrations.keys())
        self.combo_isotopes.setCurrentText(current_isotope)
        self.previous_index = self.combo_isotopes.currentIndex()
        self.combo_isotopes.currentIndexChanged.connect(self.comboChanged)

        # Check all
        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")

        # Form layout for line edits
        self.layout_form.addRow("Gradient:", self.lineedit_gradient)
        self.layout_form.addRow("Intercept:", self.lineedit_intercept)
        self.layout_form.addRow("Unit:", self.lineedit_unit)
        self.layout().insertWidget(1, self.combo_isotopes, 1, QtCore.Qt.AlignRight)
        self.layout().insertWidget(2, self.check_all)

        self.updateLineEdits()

    def updateLineEdits(self) -> None:
        name = self.combo_isotopes.currentText()

        gradient = self.calibrations[name].gradient
        if gradient == 1.0:
            self.lineedit_gradient.clear()
        else:
            self.lineedit_gradient.setText(str(gradient))
        intercept = self.calibrations[name].intercept
        if intercept == 0.0:
            self.lineedit_intercept.clear()
        else:
            self.lineedit_intercept.setText(str(intercept))
        unit = self.calibrations[name].unit
        if unit is None:
            self.lineedit_unit.clear()
        else:
            self.lineedit_unit.setText(str(unit))

    def updateCalibration(self, name: str) -> None:
        gradient = self.lineedit_gradient.text()
        intercept = self.lineedit_intercept.text()
        unit = self.lineedit_unit.text()

        if gradient != "":
            self.calibrations[name].gradient = float(gradient)
        if intercept != "":
            self.calibrations[name].intercept = float(intercept)
        if unit != "":
            self.calibrations[name].unit = unit

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
