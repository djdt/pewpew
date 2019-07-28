from PySide2 import QtCore, QtWidgets
import numpy as np

from laserlib.krisskross.data import krisskross_layers
from pewpew.ui.validators import DecimalValidator

from pewpew.ui.canvas.laser import LaserCanvas
from pewpew.ui.tools.tool import Tool

from pewpew.ui.docks.dockarea import DockArea
from pewpew.ui.docks import LaserImageDock

from typing import Callable, List


class OperationsTool(Tool):
    SYMBOLS = {
        "add": "+",
        "divide": "/",
        "true_divide": "/",
        "multiply": "*",
        "subtract": "-",
        "greater": ">",
        "less": "<",
        "equal": "=",
        "not_equal": "!=",
        "where": "=>",
    }
    OPERATIONS = {
        # Name: callable, symbol, num data
        "None": None,
        "Add": np.add,
        "Divide": np.divide,
        "Multiply": np.multiply,
        "Subtract": np.subtract,
        "Where": np.where,
    }

    CONDITIONS = {
        "None": None,
        "Equal": np.equal,
        "Greater than": np.greater,
        "Less than": np.less,
        "Not equal": np.not_equal,
    }

    def __init__(
        self,
        dockarea: DockArea,
        dock: LaserImageDock,
        viewconfig: dict,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Operations Tool")

        self.dockarea = dockarea

        self.dock = dock
        self.data = np.zeros_like((1, 1), dtype=float)
        self.fill_value = 0.0
        self.name = ""

        self.button_laser = QtWidgets.QPushButton("Select &Image...")
        self.button_laser.pressed.connect(self.buttonLaser)

        options = {"colorbar": False, "scalebar": False, "label": False}
        self.canvas = LaserCanvas(viewconfig=viewconfig, options=options, parent=self)

        self.combo_isotope1 = QtWidgets.QComboBox()
        self.combo_isotope1.currentIndexChanged.connect(self.updateData)
        self.combo_isotope1.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_condition1 = QtWidgets.QComboBox()
        self.combo_condition1.addItems(list(OperationsTool.CONDITIONS))
        self.lineedit_condition1 = QtWidgets.QLineEdit()
        self.lineedit_condition1.setValidator(DecimalValidator(-1e9, 1e9, 4))
        self.lineedit_condition1.editingFinished.connect(self.updateData)

        self.combo_ops = QtWidgets.QComboBox()
        self.combo_ops.addItems(list(OperationsTool.OPERATIONS))
        self.combo_ops.currentIndexChanged.connect(self.onComboOps)

        self.combo_isotope2 = QtWidgets.QComboBox()
        self.combo_isotope2.currentIndexChanged.connect(self.updateData)
        self.combo_isotope2.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_condition2 = QtWidgets.QComboBox()
        self.combo_condition2.addItems(list(OperationsTool.CONDITIONS))
        self.combo_condition2.currentIndexChanged.connect(self.updateData)
        self.lineedit_condition2 = QtWidgets.QLineEdit()
        self.lineedit_condition2.setValidator(DecimalValidator(-1e9, 1e9, 4))
        self.lineedit_condition2.editingFinished.connect(self.updateData)

        self.updateComboIsotopes()
        self.combo_condition1.currentIndexChanged.connect(self.updateData)
        self.combo_isotope2.currentIndexChanged.connect(self.updateData)

        # Layouts
        group_box_var1 = QtWidgets.QGroupBox()
        layout_box_var1 = QtWidgets.QFormLayout()
        layout_cond1 = QtWidgets.QHBoxLayout()
        layout_cond1.addWidget(self.combo_condition1)
        layout_cond1.addWidget(self.lineedit_condition1)
        layout_box_var1.addRow("Variable:", self.combo_isotope1)
        layout_box_var1.addRow("Condition:", layout_cond1)
        group_box_var1.setLayout(layout_box_var1)

        group_box_ops = QtWidgets.QGroupBox()
        layout_box_ops = QtWidgets.QFormLayout()
        layout_box_ops.addRow("Operation:", self.combo_ops)
        group_box_ops.setLayout(layout_box_ops)

        group_box_var2 = QtWidgets.QGroupBox()
        layout_box_var2 = QtWidgets.QFormLayout()
        layout_cond2 = QtWidgets.QHBoxLayout()
        layout_cond2.addWidget(self.combo_condition2)
        layout_cond2.addWidget(self.lineedit_condition2)
        layout_box_var2.addRow("Variable:", self.combo_isotope2)
        layout_box_var2.addRow("Condition:", layout_cond2)
        group_box_var2.setLayout(layout_box_var2)

        self.layout_top.addWidget(self.button_laser, 1, QtCore.Qt.AlignRight)

        self.layout_left.addWidget(group_box_var1, 0)
        self.layout_left.addWidget(group_box_ops, 0)
        self.layout_left.addWidget(group_box_var2, 0)
        self.layout_left.addStretch(1)

        self.layout_right.addWidget(self.canvas)
        self.updateData()
        self.onComboOps()

    # def generateName(self, c1, c1v, c2, c2v, op) -> None:
    #     name = self.combo_isotope1.currentText()
    #     c1 = OperationsTool.CONDITIONS[self.combo_condition1.currentText()]
    #     if c1 is not None:
    #         name += f"[{OperationsTool.SYMBOLS[c1.__name__]}{self.lineedit_condition1.text()}]"
    #     if self.combo_isotope2.isEnabled():
    #         op = OperationsTool.OPERATIONS[self.combo_ops.currentText()]
    #         if op is not None:
    #             name += f"{OperationsTool.SYMBOLS[op.__name__]}"

    def getMaskedName(
        self, isotope: str, condition: Callable = None, value: float = None
    ) -> str:
        name = isotope
        if condition is not None:
            name += f"[{OperationsTool.SYMBOLS[condition.__name__]}{value}]"
        return name

    def getMaskedData(
        self, isotope: str, condition: Callable = None, value: float = None
    ) -> np.ndarray:
        data = self.dock.laser.get(isotope, calibrate=False)
        if condition is not None and value is not None:
            mask = condition(data, value)
            data = np.where(mask, data, np.full_like(data, self.fill_value))
        return data

    def performOperation(
        self, op: Callable, data1: np.ndarray, data2: np.ndarray
    ) -> np.ndarray:
        if op is np.where:
            data = np.where(data2 != self.fill_value, data1, data2)
        else:
            # If true divide is used avoid errors by replacing with one, then zero
            if op == np.true_divide:
                data2[data2 == 0.0] = 1.0
            data = op(data1, data2)
            if op == np.true_divide:
                data[data2 == 0.0] = 0.0
        return data

    def updateData(self) -> None:
        isotope = self.combo_isotope1.currentText()

        if isotope not in self.dock.laser.data:
            return

        c1 = OperationsTool.CONDITIONS[self.combo_condition1.currentText()]
        try:
            c1v = float(self.lineedit_condition1.text())
        except ValueError:
            c1v = None
        data1 = self.getMaskedData(isotope, c1, c1v)
        name = self.getMaskedName(isotope, c1, c1v)

        op = OperationsTool.OPERATIONS[self.combo_ops.currentText()]

        if self.combo_isotope2.isEnabled() and op is not None:
            name += f"{OperationsTool.SYMBOLS[op.__name__]}"
            isotope2 = self.combo_isotope2.currentText()
            c2 = OperationsTool.CONDITIONS[self.combo_condition2.currentText()]
            try:
                c2v = float(self.lineedit_condition2.text())
            except ValueError:
                c2v = None
            data2 = self.getMaskedData(isotope2, c2, c2v)
            name += self.getMaskedName(isotope2, c2, c2v)

            self.data = self.performOperation(op, data1, data2)
        else:
            self.data = data1

        self.name = name
        self.draw()

    def draw(self) -> None:
        self.canvas.drawData(self.data, self.dock.laser.config.data_extent(self.data))
        self.canvas.draw()

    def updateComboIsotopes(self) -> None:
        self.combo_isotope1.blockSignals(True)
        self.combo_isotope2.blockSignals(True)
        isotope1 = self.combo_isotope1.currentText()
        self.combo_isotope1.clear()
        self.combo_isotope1.addItems(self.dock.laser.isotopes())
        self.combo_isotope1.setCurrentText(isotope1)

        isotope2 = self.combo_isotope2.currentText()
        self.combo_isotope2.clear()
        self.combo_isotope2.addItems(self.dock.laser.isotopes())
        self.combo_isotope2.setCurrentText(isotope2)
        self.combo_isotope1.blockSignals(False)
        self.combo_isotope2.blockSignals(False)

    def onComboOps(self) -> None:
        enabled = self.combo_ops.currentText() != "None"
        self.combo_isotope2.setEnabled(enabled)
        self.combo_condition2.setEnabled(enabled)
        self.lineedit_condition2.setEnabled(enabled)

        self.updateData()

    @QtCore.Slot("QWidget*")
    def mouseSelectFinished(self, widget: QtWidgets.QWidget) -> None:
        if widget is not None and hasattr(widget, "laser"):
            self.dock = widget
            # Prevent currentIndexChanged being emmited
            self.updateComboIsotopes()
            self.updateData()

        self.dockarea.mouseSelectFinished.disconnect(self.mouseSelectFinished)
        self.activateWindow()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.show()
        self.draw()

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() in [
            QtCore.Qt.Key_Escape,
            QtCore.Qt.Key_Enter,
            QtCore.Qt.Key_Return,
        ]:
            return
        if event.key() == QtCore.Qt.Key_F5:
            self.draw()
        super().keyPressEvent(event)

    def buttonLaser(self) -> None:
        self.hide()
        self.dockarea.activateWindow()
        self.dockarea.setFocus(QtCore.Qt.OtherFocusReason)
        self.dockarea.startMouseSelect()
        self.dockarea.mouseSelectFinished.connect(self.mouseSelectFinished)


class KrissKrossOperationsTool(OperationsTool):
    def getMaskedData(
        self, isotope: str, condition: Callable = None, value: float = None
    ) -> List[np.ndarray]:
        data = self.dock.laser.data[isotope].data[:]
        if condition is not None and value is not None:
            for i, d in enumerate(data):
                mask = condition(d, value)
                data[i] = np.where(mask, d, np.full_like(d, self.fill_value))
        return data

    def performOperation(
        self, op: Callable, data1: List[np.ndarray], data2: List[np.ndarray]
    ) -> List[np.ndarray]:
        if op is np.where:
            data = [
                np.where(d2 != self.fill_value, d1, d2) for d1, d2 in zip(data1, data2)
            ]
        else:
            # If true divide is used avoid errors by replacing with one, then zero
            if op == np.true_divide:
                for d in data2:
                    d[d == 0] = 1.0
            data = [op(d1, d2) for d1, d2 in zip(data1, data2)]
            if op == np.true_divide:
                for d in data2:
                    d[d == 0] = 0.0
        return data

    def draw(self) -> None:
        # Assemble data
        data = np.mean(
            krisskross_layers(self.data, self.dock.laser.config), axis=2  # type: ignore
        )
        self.canvas.drawData(data, self.dock.laser.config.data_extent(data))
        self.canvas.draw()
