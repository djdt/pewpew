from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pewpew.ui.widgets import BasicTable, Canvas

# from pewpew.ui.dialogs import ApplyDialog
from pewpew.ui.tools.tool import Tool
from pewpew.ui.validators import DoublePrecisionDelegate

from typing import Dict, List
from pewpew.ui.docks.dockarea import DockArea
from pewpew.ui.docks import LaserImageDock


class CalculationsTool(Tool):
    OPERATIONS = {"Add", (np.add, "+")}
    CONDITIONS = {"Greater than", (np.greater, ">")}

    def __init__(
        self,
        dock: LaserImageDock,
        dockarea: DockArea,
        viewconfig: dict,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration Standards Tool")

        self.dockarea = dockarea
        self.viewconfig = viewconfig

        self.dock = dock
        self.button_laser = QtWidgets.QPushButton("Select &Image...")

        self.canvas = Canvas(parent=self)
        self.canvas.options = {"colorbar": False, "scalebar": False, "label": False}

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.addItems(self.dock.laser.isotopes())
        self.combo_isotope.currentIndexChanged.connect(self.comboIsotope)

        group_box_var1 = QtWidgets.QGroupBox()
        layout_box_var1 = QtWidgets.QFormLayout()
        self.combo_isotope1 = QtWidgets.QComboBox()
        self.combo_isotope1.addItems(self.dock.laser.isotopes())
        self.combo_condition1 = QtWidgets.QComboBox()
        self.combo_condition1.addItems(CalculationsTool.CONDITIONS)
        group_box_var1.setLayout(layout_box_var1)

        group_box_ops = QtWidgets.QGroupBox()
        layout_box_ops = QtWidgets.QFormLayout()
        self.combo_ops = QtWidgets.QComboBox()
        self.combo_ops.addItems(CalculationsTool.OPERATIONS)
        group_box_ops.setLayout(layout_box_ops)

        group_box_var2 = QtWidgets.QGroupBox()
        layout_box_var2 = QtWidgets.QFormLayout()
        self.combo_isotope2 = QtWidgets.QComboBox()
        self.combo_isotope2.addItems(self.dock.laser.isotopes())
        self.combo_condition2 = QtWidgets.QComboBox()
        self.combo_condition2.addItems(CalculationsTool.CONDITIONS)
        group_box_var2.setLayout(layout_box_var2)

    def accept(self) -> None:
        self.updateCalculations()
        self.applyPressed.emit(self)
        super().accept()

    def apply(self) -> None:
        pass

    def draw(self) -> None:
        self.canvas.clear()
        if self.combo_isotope.currentText() in self.dock.laser.data:
            self.canvas.plot(
                self.dock.laser, self.combo_isotope.currentText(), self.viewconfig
            )
            self.canvas.plotLevels(self.spinbox_levels.value())
            self.canvas.draw()

    def updateCalibration(self) -> None:
        pass
        # self.dock.laser.calibration = self.calibration
        # self.dock.draw()

    @QtCore.pyqtSlot("QWidget*")
    def mouseSelectFinished(self, widget: QtWidgets.QWidget) -> None:
        if widget is not None and hasattr(widget, "laser"):
            self.dock = widget
            self.calibration = {
                k: (v.gradient, v.intercept, v.unit)
                for k, v in widget.laser.data.items()
            }
            # Prevent currentIndexChanged being emmited
            self.combo_isotope.blockSignals(True)
            self.combo_isotope.clear()
            self.combo_isotope.addItems(self.dock.laser.isotopes())
            self.combo_isotope.blockSignals(False)

            self.lineedit_left.setText("")
            self.lineedit_right.setText("")

            self.updateCounts()
            self.updateResults()

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

    def comboIsotope(self, text: str) -> None:
        isotope = self.combo_isotope.currentText()
        texts = self.table.columnText(CalibrationTable.COLUMN_CONC)
        # Only update if at least one cell is filled
        for text in texts:
            if text != "":
                self.texts[self.previous_isotope] = texts
                break

        self.updateConcentrations()
        self.updateCounts()
        self.updateResults()
        self.draw()
        self.previous_isotope = isotope
