from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np
import copy

from laserlib.calibration import LaserCalibration

from pewpew.ui.tools.tool import Tool
from pewpew.ui.docks.dockarea import DockArea
from pewpew.ui.docks.laserdock import LaserImageDock
from pewpew.ui.dialogs.calibrationcurve import CalibrationCurveDialog

from pewpew.ui.tools.standards.canvas import StandardsCanvas
from pewpew.ui.tools.standards.results import StandardsResultsBox
from pewpew.ui.tools.standards.table import StandardsTable


class StandardsTool(Tool):
    def __init__(
        self,
        dockarea: DockArea,
        dock: LaserImageDock,
        viewconfig: dict,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration Standards Tool")

        self.dockarea = dockarea
        self.previous_isotope = ""

        self.dock = dock
        self.calibrations = {
            k: copy.copy(v.calibration) for k, v in self.dock.laser.data.items()
        }

        # Left side
        self.spinbox_levels = QtWidgets.QSpinBox()
        self.spinbox_levels.setMinimum(1)
        self.spinbox_levels.setMaximum(20)
        self.spinbox_levels.setValue(6)
        self.spinbox_levels.valueChanged.connect(self.spinBoxLevels)

        self.lineedit_units = QtWidgets.QLineEdit()
        self.lineedit_units.editingFinished.connect(self.lineeditUnits)

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_weighting.addItems(["None", "x", "1/x", "1/(x^2)"])
        self.combo_weighting.currentIndexChanged.connect(self.comboWeighting)

        self.results_box = StandardsResultsBox()
        self.results_box.button.pressed.connect(self.showCurve)

        # Right side
        self.button_laser = QtWidgets.QPushButton("Select &Image...")
        self.button_laser.pressed.connect(self.buttonLaser)

        self.canvas = StandardsCanvas(viewconfig, parent=self)

        self.lineedit_left = QtWidgets.QLineEdit()
        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_left.editingFinished.connect(self.lineEditTrim)
        self.lineedit_right = QtWidgets.QLineEdit()
        self.lineedit_right.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_right.editingFinished.connect(self.lineEditTrim)

        self.combo_trim = QtWidgets.QComboBox()
        self.combo_trim.addItems(["rows", "s", "μm"])
        self.combo_trim.setCurrentIndex(1)
        self.combo_trim.currentIndexChanged.connect(self.comboTrim)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.addItems(sorted(self.dock.laser.isotopes()))
        self.combo_isotope.setCurrentText(self.dock.combo_isotope.currentText())
        self.combo_isotope.currentIndexChanged.connect(self.comboIsotope)

        isotope = self.combo_isotope.currentText()
        self.table = StandardsTable(
            self.calibrations[isotope] if isotope != "" else LaserCalibration(), self
        )
        self.table.setRowCount(6)
        self.table.model().dataChanged.connect(self.updateResults)

        self.layoutWidgets()
        self.combo_weighting.setCurrentText(self.calibrations[isotope].weighting)
        self.lineedit_units.setText(self.calibrations[isotope].unit)
        self.draw()
        self.updateCounts()

    def layoutWidgets(self) -> None:
        layout_cal_form = QtWidgets.QFormLayout()
        layout_cal_form.addRow("Calibration Levels:", self.spinbox_levels)

        layout_table_form = QtWidgets.QFormLayout()
        layout_table_form.addRow("Units:", self.lineedit_units)
        layout_table_form.addRow("Weighting:", self.combo_weighting)

        self.layout_left.addLayout(layout_cal_form)
        self.layout_left.addWidget(self.table)
        self.layout_left.addLayout(layout_table_form)
        self.layout_left.addWidget(self.results_box)

        layout_box_trim = QtWidgets.QHBoxLayout()
        layout_box_trim.addWidget(QtWidgets.QLabel("Left:"))
        layout_box_trim.addWidget(self.lineedit_left)
        layout_box_trim.addWidget(QtWidgets.QLabel("Right:"))
        layout_box_trim.addWidget(self.lineedit_right)
        layout_box_trim.addWidget(self.combo_trim)

        box_trim = QtWidgets.QGroupBox("Trim")
        box_trim.setLayout(layout_box_trim)

        layout_canvas_bar = QtWidgets.QHBoxLayout()
        layout_canvas_bar.addWidget(box_trim)
        layout_canvas_bar.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignTop)

        self.layout_top.addWidget(self.button_laser, 0, QtCore.Qt.AlignRight)
        self.layout_right.addWidget(self.canvas)
        self.layout_right.addLayout(layout_canvas_bar)

    def draw(self) -> None:
        isotope = self.combo_isotope.currentText()
        if isotope in self.dock.laser.data:
            self.canvas.drawLaser(self.dock.laser, isotope)
            self.canvas.drawLevels(
                StandardsTable.ROW_LABELS, self.spinbox_levels.value()
            )
            self.canvas.draw()

    def changeCalibration(self) -> None:
        isotope = self.combo_isotope.currentText()
        self.table.model().setCalibration(self.calibrations[isotope])

    def updateCounts(self) -> None:
        isotope = self.combo_isotope.currentText()
        data = self.dock.laser.get(
            isotope, calibrate=False, extent=self.canvas.view_limits
        )
        if len(data) == 1:
            return
        buckets = np.array_split(data, self.spinbox_levels.value(), axis=0)
        self.table.setCounts([np.mean(b) for b in buckets])

    def updateResults(self) -> None:
        # Clear results if not complete
        if not self.table.isComplete():
            self.results_box.clear()
            return
        else:
            isotope = self.combo_isotope.currentText()
            self.results_box.update(self.calibrations[isotope])

    @QtCore.Slot("QWidget*")
    def mouseSelectFinished(self, widget: QtWidgets.QWidget) -> None:
        if widget is not None and hasattr(widget, "laser"):
            self.dock = widget
            self.calibrations = {
                k: copy.copy(v.calibration) for k, v in widget.laser.data.items()
            }
            # Prevent currentIndexChanged being emmited
            self.combo_isotope.blockSignals(True)
            self.combo_isotope.clear()
            self.combo_isotope.addItems(sorted(self.dock.laser.isotopes()))
            self.combo_isotope.setCurrentText(self.dock.combo_isotope.currentText())
            self.combo_isotope.blockSignals(False)

            self.lineedit_left.setText("")
            self.lineedit_right.setText("")

            isotope = self.combo_isotope.currentText()
            self.combo_weighting.setCurrentText(
                self.calibrations[isotope].weighting
            )
            self.lineedit_units.setText(self.calibrations[isotope].unit)
            self.draw()
            self.changeCalibration()
            self.updateCounts()
            self.updateResults()

        self.dockarea.mouseSelectFinished.disconnect(self.mouseSelectFinished)
        self.activateWindow()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.show()

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() in [
            QtCore.Qt.Key_Escape,
            QtCore.Qt.Key_Enter,
            QtCore.Qt.Key_Return,
        ]:
            return
        elif event.key() == QtCore.Qt.Key_F5:
            self.draw()
        super().keyPressEvent(event)

    def showCurve(self) -> None:
        dlg = CalibrationCurveDialog(
            self.calibrations[self.combo_isotope.currentText()], parent=self
        )
        dlg.show()

    def buttonLaser(self) -> None:
        self.hide()
        self.dockarea.activateWindow()
        self.dockarea.setFocus(QtCore.Qt.OtherFocusReason)
        self.dockarea.startMouseSelect()
        self.dockarea.mouseSelectFinished.connect(self.mouseSelectFinished)

    def comboAveraging(self, text: str) -> None:
        self.updateCounts()

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

        self.changeCalibration()
        if self.calibrations[isotope].unit != "":
            self.lineedit_units.setText(self.calibrations[isotope].unit)
        else:
            self.calibrations[isotope].unit = self.lineedit_units.text()
        if self.calibrations[isotope].weighting is not None:
            self.combo_weighting.setCurrentText(self.calibrations[isotope].weighting)
        else:
            self.calibrations[isotope].weighting = self.combo_weighting.currentText()
        self.draw()
        self.updateCounts()

    def comboWeighting(self, text: str) -> None:
        isotope = self.combo_isotope.currentText()
        self.calibrations[isotope].weighting = self.combo_weighting.currentText()
        self.calibrations[isotope].update_linreg()
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
        self.canvas.view_limits = (
            trim_left,
            trim_right,
            0.0,
            self.canvas.image.get_extent()[3],
        )
        self.canvas.updateView()
        self.updateCounts()

    def lineeditUnits(self) -> None:
        isotope = self.combo_isotope.currentText()
        unit = self.lineedit_units.text()
        self.calibrations[isotope].unit = unit

    def spinBoxLevels(self) -> None:
        self.table.setRowCount(self.spinbox_levels.value())
        self.updateCounts()
        self.draw()
