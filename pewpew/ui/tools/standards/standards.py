from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import copy

from laserlib.calibration import LaserCalibration

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter

from pewpew.ui.tools.tool import Tool
from pewpew.ui.docks.dockarea import DockArea
from pewpew.ui.docks.laserdock import LaserImageDock

from pewpew.ui.tools.standards.canvas import StandardsCanvas
from pewpew.ui.tools.standards.results import (
    StandardsResultsBox,
    StandardsResultsDialog,
)
from pewpew.ui.tools.standards.table import StandardsTable

from typing import Dict, List


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
        self.texts: Dict[str, List[str]] = {}

        # Left side
        self.spinbox_levels = QtWidgets.QSpinBox()

        self.table = StandardsTable(self)

        self.lineedit_units = QtWidgets.QLineEdit()

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_averaging = QtWidgets.QComboBox()

        self.results_box = StandardsResultsBox()

        # Right side
        self.button_laser = QtWidgets.QPushButton("Select &Image...")

        self.canvas = StandardsCanvas(viewconfig, parent=self)

        self.lineedit_left = QtWidgets.QLineEdit()
        self.lineedit_right = QtWidgets.QLineEdit()

        self.combo_trim = QtWidgets.QComboBox()

        self.combo_isotope = QtWidgets.QComboBox()

        self.initialiseWidgets()
        self.layoutWidgets()

        self.previous_isotope = self.combo_isotope.currentText()

        self.draw()
        self.updateCounts()

    def initialiseWidgets(self) -> None:
        self.spinbox_levels.setMinimum(1)
        self.spinbox_levels.setMaximum(20)
        self.spinbox_levels.setValue(6)
        self.spinbox_levels.valueChanged.connect(self.spinBoxLevels)

        self.lineedit_units.editingFinished.connect(self.lineeditUnits)

        self.combo_weighting.addItems(["None", "x", "1/x", "1/(x^2)"])
        self.combo_weighting.currentIndexChanged.connect(self.comboWeighting)

        self.combo_averaging.addItems(["Mean", "Median"])
        self.combo_averaging.currentIndexChanged.connect(self.comboAveraging)

        self.results_box.button.pressed.connect(self.showCurve)

        self.table.setRowCount(6)
        self.table.itemChanged.connect(self.tableItemChanged)

        self.button_laser.pressed.connect(self.buttonLaser)

        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_left.editingFinished.connect(self.lineEditTrim)
        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_right.editingFinished.connect(self.lineEditTrim)

        self.combo_trim.addItems(["rows", "s", "μm"])
        self.combo_trim.setCurrentIndex(1)
        self.combo_trim.currentIndexChanged.connect(self.comboTrim)

        self.combo_isotope.addItems(self.dock.laser.isotopes())
        self.combo_isotope.setCurrentText(self.dock.combo_isotope.currentText())
        self.combo_isotope.currentIndexChanged.connect(self.comboIsotope)

    def layoutWidgets(self) -> None:
        layout_cal_form = QtWidgets.QFormLayout()
        layout_cal_form.addRow("Calibration Levels:", self.spinbox_levels)

        layout_table_form = QtWidgets.QFormLayout()
        layout_table_form.addRow("Units:", self.lineedit_units)
        layout_table_form.addRow("Weighting:", self.combo_weighting)
        layout_table_form.addRow("Averaging:", self.combo_averaging)

        # layout_left = QtWidgets.QVBoxLayout()
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
        if self.combo_isotope.currentText() in self.dock.laser.data:
            self.canvas.drawLaser(self.dock.laser, self.combo_isotope.currentText())
            self.canvas.drawLevels(
                StandardsTable.ROW_LABELS, self.spinbox_levels.value()
            )
            self.canvas.draw()

    def updateConcentrations(self) -> None:
        name = self.combo_isotope.currentText()
        self.table.blockSignals(True)
        concentrations = [str(x) for x in self.calibrations[name].concentrations()]
        if len(concentrations) > 0:
            self.table.setColumnText(StandardsTable.COLUMN_CONC, concentrations)
        else:
            self.table.setColumnText(StandardsTable.COLUMN_CONC, None)
        self.table.blockSignals(False)

    def updateCounts(self) -> None:
        data = self.dock.laser.get(
            self.combo_isotope.currentText(),
            calibrate=False,
            extent=self.canvas.view_limits,
        )
        if len(data) == 1:
            return

        if self.canvas.viewconfig["filtering"]["type"] != "None":
            filter_type, window, threshold = (
                self.canvas.viewconfig["filtering"][x]
                for x in ["type", "window", "threshold"]
            )
            data = data.copy()
            if filter_type == "Rolling mean":
                rolling_mean_filter(data, window, threshold)
            elif filter_type == "Rolling median":
                rolling_median_filter(data, window, threshold)

        sections = np.array_split(data, self.table.rowCount(), axis=0)
        text = []
        averging = self.combo_averaging.currentText()
        for row in range(0, self.table.rowCount()):
            if averging == "Median":
                text.append(f"{np.median(sections[row])}")
            else:  # Mean
                text.append(f"{np.mean(sections[row])}")

        self.table.blockSignals(True)
        self.table.setColumnText(StandardsTable.COLUMN_COUNT, text)
        self.table.blockSignals(False)

    def updateResults(self) -> None:
        # Clear results if not complete
        if not self.table.isComplete():
            self.results_box.clear()
            return

        x = np.array(
            [
                x if x != "" else "nan"
                for x in self.table.columnText(StandardsTable.COLUMN_CONC)
            ],
            dtype=np.float64,
        )
        y = np.array(
            self.table.columnText(StandardsTable.COLUMN_COUNT), dtype=np.float64
        )
        # Strip negative x values
        points = np.stack([x, y], axis=1)

        name = self.combo_isotope.currentText()
        self.calibrations[name] = LaserCalibration.from_points(
            points=points,
            weighting=self.combo_weighting.currentText(),
            unit=self.lineedit_units.text(),
        )
        self.results_box.update(self.calibrations[name])

    @QtCore.pyqtSlot("QWidget*")
    def mouseSelectFinished(self, widget: QtWidgets.QWidget) -> None:
        if widget is not None and hasattr(widget, "laser"):
            self.dock = widget
            self.calibrations = {
                k: copy.copy(v.calibration) for k, v in widget.laser.data.items()
            }
            # Prevent currentIndexChanged being emmited
            self.combo_isotope.blockSignals(True)
            self.combo_isotope.clear()
            self.combo_isotope.addItems(self.dock.laser.isotopes())
            self.combo_isotope.setCurrentText(self.dock.combo_isotope.currentText())
            self.combo_isotope.blockSignals(False)

            self.lineedit_left.setText("")
            self.lineedit_right.setText("")

            self.updateCounts()
            self.updateResults()

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
        elif event.key() == QtCore.Qt.Key_F5:
            self.draw()
        super().keyPressEvent(event)

    def showCurve(self) -> None:
        dlg = StandardsResultsDialog(
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
        texts = self.table.columnText(StandardsTable.COLUMN_CONC)
        # Only update if at least one cell is filled
        for text in texts:
            if text != "":
                self.texts[self.previous_isotope] = texts
                break

        if isotope in self.calibrations:
            self.lineedit_units.setText(self.calibrations[isotope].unit)

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
        self.canvas.view_limits = (
            trim_left,
            trim_right,
            0.0,
            self.canvas.image.get_extent()[3],
        )
        self.canvas.updateView()
        self.updateCounts()
        self.updateResults()

    def lineeditUnits(self) -> None:
        unit = self.lineedit_units.text()
        self.calibrations[self.combo_isotope.currentText()].unit = unit

    def spinBoxLevels(self) -> None:
        self.table.setRowCount(self.spinbox_levels.value())
        self.updateCounts()
        self.updateResults()
        self.draw()

    def tableItemChanged(self, item: QtWidgets.QTableWidgetItem) -> None:
        self.updateResults()
