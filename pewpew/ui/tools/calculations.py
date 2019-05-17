from PyQt5 import QtCore, QtWidgets
import numpy as np

from laserlib.laser import Laser
from pewpew.ui.validators import DecimalValidator

from pewpew.ui.widgets import Canvas
from pewpew.ui.tools.tool import Tool

from pewpew.ui.docks.dockarea import DockArea
from pewpew.ui.docks import LaserImageDock


class CalculationsTool(Tool):
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
        dock: LaserImageDock,
        dockarea: DockArea,
        viewconfig: dict,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calculations Tool")

        self.dockarea = dockarea

        self.dock = dock
        self.data = np.zeros_like((1, 1), dtype=float)
        self.fill_value = -1.0
        self.name = ""

        self.button_laser = QtWidgets.QPushButton("Select &Image...")

        options = {"colorbar": False, "scalebar": False, "label": False}
        self.canvas = Canvas(
            viewconfig=viewconfig,
            options=options,
            connect_mouse_events=False,
            parent=self,
        )

        self.combo_isotope1 = QtWidgets.QComboBox()
        self.combo_isotope1.currentIndexChanged.connect(self.updateData)
        self.combo_isotope1.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_condition1 = QtWidgets.QComboBox()
        self.combo_condition1.addItems(CalculationsTool.CONDITIONS)
        self.lineedit_condition1 = QtWidgets.QLineEdit()
        self.lineedit_condition1.setValidator(DecimalValidator(-1e9, 1e9, 4))
        self.lineedit_condition1.editingFinished.connect(self.updateData)

        self.combo_ops = QtWidgets.QComboBox()
        self.combo_ops.addItems(CalculationsTool.OPERATIONS)
        self.combo_ops.currentIndexChanged.connect(self.onComboOps)

        self.combo_isotope2 = QtWidgets.QComboBox()
        self.combo_isotope2.currentIndexChanged.connect(self.updateData)
        self.combo_isotope2.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_condition2 = QtWidgets.QComboBox()
        self.combo_condition2.addItems(CalculationsTool.CONDITIONS)
        self.combo_condition2.currentIndexChanged.connect(self.updateData)
        self.lineedit_condition2 = QtWidgets.QLineEdit()
        self.lineedit_condition2.setValidator(DecimalValidator(-1e9, 1e9, 4))
        self.lineedit_condition2.editingFinished.connect(self.updateData)

        self.updateComboIsotopes()
        self.combo_condition1.currentIndexChanged.connect(self.updateData)
        self.combo_isotope2.currentIndexChanged.connect(self.updateData)
        self.onComboOps()

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

    def updateData(self) -> None:
        isotope = self.combo_isotope1.currentText()

        if isotope not in self.dock.laser.data:
            return

        data = self.dock.laser.data[isotope].data
        name = isotope

        c1 = CalculationsTool.CONDITIONS[self.combo_condition1.currentText()]
        if c1 is not None:
            try:
                c1v = float(self.lineedit_condition1.text())
                mask = c1(data, c1v)
                data = np.where(mask, data, np.full_like(data, self.fill_value))
                name += f"[{CalculationsTool.SYMBOLS[c1.__name__]}{c1v}]"
            except ValueError:
                pass

        if self.combo_isotope2.isEnabled():
            op = CalculationsTool.OPERATIONS[self.combo_ops.currentText()]
            if op is not None:
                name += f"{CalculationsTool.SYMBOLS[op.__name__]}"

            data2 = self.dock.laser.data[self.combo_isotope2.currentText()].data
            name += self.combo_isotope2.currentText()

            c2 = CalculationsTool.CONDITIONS[self.combo_condition2.currentText()]
            if c2 is not None:
                try:
                    c2v = float(self.lineedit_condition2.text())
                    mask = c2(data2, c2v)
                    data2 = np.where(mask, data2, np.full_like(data2, self.fill_value))
                    name += f"[{CalculationsTool.SYMBOLS[c2.__name__]}{c2v}]"
                except ValueError:
                    pass

            if op is np.where:
                data = np.where(data2 != self.fill_value, data, data2)
            elif op is not None:
                # If true divide is used we avoid errors by replacing with one, then zero
                if op == np.true_divide:
                    data2[data2 == 0.0] = 1.0
                data = op(data, data2)
                if op == np.true_divide:
                    data[data2 == 0.0] = 0.0

        self.data = data
        self.name = name

        self.draw()

    def draw(self) -> None:
        self.canvas.drawData(
            self.data,
            self.dock.laser.config.data_extent(self.data),
            self.dock.laser.config.aspect(),
        )
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

    @QtCore.pyqtSlot("QWidget*")
    def mouseSelectFinished(self, widget: QtWidgets.QWidget) -> None:
        if widget is not None and hasattr(widget, "laser"):
            self.dock = widget
            self.laser = Laser(config=self.dock.laser.config)
            # Prevent currentIndexChanged being emmited
            self.updateComboIsotopes()
            self.updateData()

        self.dockarea.mouseSelectFinished.disconnect(self.mouseSelectFinished)
        self.activateWindow()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.canvas.view = (0.0, 0.0, 0.0, 0.0)
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
