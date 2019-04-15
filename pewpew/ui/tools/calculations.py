from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pewpew.ui.widgets import BasicTable, Canvas
# from pewpew.ui.dialogs import ApplyDialog
from pewpew.ui.tools.tool import Tool
from pewpew.ui.validators import DoublePrecisionDelegate

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter, weighted_linreg

from typing import Dict, List
from pewpew.ui.docks.dockarea import DockArea
from pewpew.ui.docks import LaserImageDock


class CalculationsTool(Tool):
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
        self.previous_isotope = ""

        self.dock = dock
        self.button_laser = QtWidgets.QPushButton("Select &Image...")
        self.canvas = CalibrationCanvas(parent=self)
        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.addItems(self.dock.laser.isotopes())
        self.combo_isotope.currentIndexChanged.connect(self.comboIsotope)

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
        if event.key() in [QtCore.Qt.Key_Escape, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
            return
        if event.key() == QtCore.Qt.Key_F5:
            self.draw()
        super().keyPressEvent(event)

    def buttonBoxClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.Ok:
            self.accept()
        else:
            self.reject()

    def buttonLaser(self) -> None:
        self.hide()
        self.dockarea.activateWindow()
        self.dockarea.setFocus(QtCore.Qt.OtherFocusReason)
        self.dockarea.startMouseSelect()
        self.dockarea.mouseSelectFinished.connect(self.mouseSelectFinished)

    def comboAveraging(self, text: str) -> None:
        self.updateCounts()
        self.updateResults()

    def comboTrim(self, text: str) -> None:
        if self.combo_trim.currentText() == "rows":
            self.lineedit_left.setValidator(QtGui.QIntValidator(0, 1e9))
            self.lineedit_right.setValidator(QtGui.QIntValidator(0, 1e9))
        else:
            self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
            self.lineedit_right.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_left.setText("")
        self.lineedit_right.setText("")

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

    def comboWeighting(self, text: str) -> None:
        self.updateResults()

    def lineEditTrim(self) -> None:
        if self.lineedit_left.text() == "":
            trim_left = 0.0
        else:
            trim_left = self.dock.laser.convert(
                float(self.lineedit_left.text()),
                unit_from=self.combo_trim.currentText(),
                unit_to="um",
            )
        trim_right = self.canvas.image.get_extent()[1]
        if self.lineedit_right.text() != "":
            trim_right -= self.dock.laser.convert(
                float(self.lineedit_right.text()),
                unit_from=self.combo_trim.currentText(),
                unit_to="um",
            )
        self.canvas.setView(trim_left, trim_right, 0.0, self.canvas.extent[3])
        self.updateCounts()
        self.updateResults()

    def lineeditUnits(self) -> None:
        unit = self.lineedit_units.text()
        cal = self.cal[self.combo_isotope.currentText()]
        self.calibration[self.combo_isotope.currentText()] = (cal[0], cal[1], unit)

    def spinBoxLevels(self) -> None:
        self.table.setRowCount(self.spinbox_levels.value())
        self.updateCounts()
        self.updateResults()
        self.draw()

    def tableItemChanged(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.text() != "":
            self.updateResults()
