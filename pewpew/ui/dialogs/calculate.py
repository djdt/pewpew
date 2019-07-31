from PySide2 import QtWidgets
import numpy as np

from pewpew.ui.dialogs.applydialog import ApplyDialog

# from pewpew.ui.validators import DecimalValidator

from laserlib.laser import Laser
from laserlib.laser.virtual import VirtualData


class CalculateDialog(ApplyDialog):
    VALID_OPS = {
        # Name: callable, symbol, num data
        "Add": (np.add, "+", 2),
        "Divide": (np.divide, "/", 2),
        "Multiply": (np.multiply, "*", 2),
        "Subtract": (np.subtract, "-", 2),
    }

    def __init__(self, laser: Laser, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.laser = laser
        self.data: VirtualData = None  # type: ignore

        self.combo_var1 = QtWidgets.QComboBox()
        self.combo_var1.addItems(self.laser.isotopes)
        self.combo_var2 = QtWidgets.QComboBox()
        self.combo_var2.addItems(self.laser.isotopes)
        self.combo_ops = QtWidgets.QComboBox()
        self.combo_ops.addItems(list(CalculateDialog.VALID_OPS.keys()))

        # self.combo_condition = QtWidgets.QComboBox()

        self.layout_form.addRow("Variable 1:", self.combo_var1)
        self.layout_form.addRow("Operation:", self.combo_ops)
        self.layout_form.addRow("Variable 2:", self.combo_var2)

    def updateData(self) -> None:
        var1 = self.combo_var1.currentText()
        var2 = self.combo_var2.currentText()

        op = CalculateDialog.VALID_OPS[self.combo_ops.currentText()]
        d1 = self.laser.data[var1]
        d2 = self.laser.data[var2]
        self.data = VirtualData(d1, f"{var1} {op[1]} {var2}", op[0], d2)

    def apply(self) -> None:
        self.updateData()

    def accept(self) -> None:
        self.updateData()
        super().accept()
