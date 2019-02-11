from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import copy

from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pewpew.ui.widgets import BasicTable, Canvas
from pewpew.ui.dialogs import ApplyDialog
from pewpew.ui.validators import DoublePrecisionDelegate

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter, weighted_linreg

from typing import Dict, List
from pewpew.ui.docks.dockarea import DockArea
from pewpew.ui.docks import LaserImageDock


class CalibrationTable(BasicTable):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]
    COLUMN_CONC = 0
    COLUMN_COUNT = 1

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Concentration", "Counts"])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.setItemDelegate(DoublePrecisionDelegate(4))

    def isComplete(self) -> bool:
        for row in range(0, self.rowCount()):
            for column in range(0, self.columnCount()):
                if self.item(row, column).text() == "":
                    return False
        return True

    def setRowCount(self, rows: int) -> None:
        current_rows = self.rowCount()
        super().setRowCount(rows)

        if current_rows < rows:
            self.setVerticalHeaderLabels(CalibrationTable.ROW_LABELS)
            for row in range(current_rows, rows):
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.setItem(row, 0, item)
                item = QtWidgets.QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                # Non editable item
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(row, 1, item)


class CalibrationCanvas(Canvas):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(connect_mouse_events=False, parent=parent)

        self.options = {"colorbar": False, "scalebar": False, "label": False}

    def plotLevels(self, levels: int) -> None:
        ax_fraction = 1.0 / levels
        # Draw lines
        for frac in np.linspace(1.0 - ax_fraction, ax_fraction, levels - 1):
            line = Line2D(
                (0.0, 1.0),
                (frac, frac),
                transform=self.ax.transAxes,
                color="black",
                linestyle="--",
                linewidth=2.0,
            )
            self.ax.add_artist(line)

        # Bookend for text
        div = make_axes_locatable(self.ax)
        cax = div.append_axes("left", size=0.2, pad=0, sharey=self.ax)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.set_facecolor("black")

        for i, frac in enumerate(np.linspace(1.0, ax_fraction, levels)):
            text = Text(
                x=0.5,
                y=frac - (ax_fraction / 2.0),
                text=CalibrationTable.ROW_LABELS[i],
                transform=cax.transAxes,
                color="white",
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="center",
            )
            cax.add_artist(text)

        self.draw()


class CalibrationTool(ApplyDialog):
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
        self.calibration = copy.deepcopy(dock.laser.calibration)  # copy dict
        self.texts: Dict[str, List[str]] = {}

        # Left side
        self.spinbox_levels = QtWidgets.QSpinBox()

        self.table = CalibrationTable(self)

        self.lineedit_units = QtWidgets.QLineEdit()

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_averaging = QtWidgets.QComboBox()

        self.lineedit_gradient = QtWidgets.QLineEdit()
        self.lineedit_intercept = QtWidgets.QLineEdit()
        self.lineedit_rsq = QtWidgets.QLineEdit()

        # Right side
        self.button_laser = QtWidgets.QPushButton("Select &Image...")

        self.canvas = CalibrationCanvas(parent=self)

        self.lineedit_left = QtWidgets.QLineEdit()
        self.lineedit_right = QtWidgets.QLineEdit()

        self.combo_trim = QtWidgets.QComboBox()

        self.combo_isotope = QtWidgets.QComboBox()

        # self.button_box = QtWidgets.QDialogButtonBox(
        #     QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok, self
        # )

        self.initialiseWidgets()
        self.layoutWidgets()

        self.previous_isotope = self.combo_isotope.currentText()

        self.draw()
        self.updateCounts()

    def initialiseWidgets(self) -> None:
        self.spinbox_levels.setMaximum(20)
        self.spinbox_levels.setValue(5)
        self.spinbox_levels.valueChanged.connect(self.spinBoxLevels)

        self.lineedit_units.editingFinished.connect(self.lineeditUnits)

        self.combo_weighting.addItems(["None", "x", "1/x", "1/(x^2)"])
        self.combo_weighting.currentIndexChanged.connect(self.comboWeighting)

        self.combo_averaging.addItems(["Mean", "Median"])
        self.combo_averaging.currentIndexChanged.connect(self.comboAveraging)

        self.table.itemChanged.connect(self.tableItemChanged)
        self.table.setRowCount(5)

        self.lineedit_gradient.setReadOnly(True)
        self.lineedit_intercept.setReadOnly(True)
        self.lineedit_rsq.setReadOnly(True)

        self.button_laser.pressed.connect(self.buttonLaser)

        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_left.editingFinished.connect(self.lineEditTrim)
        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_right.editingFinished.connect(self.lineEditTrim)

        self.combo_trim.addItems(["rows", "s", "μm"])
        self.combo_trim.setCurrentIndex(1)
        self.combo_trim.currentIndexChanged.connect(self.comboTrim)

        self.combo_isotope.addItems(self.dock.laser.isotopes())
        self.combo_isotope.currentIndexChanged.connect(self.comboIsotope)

        # self.button_box.clicked.connect(self.buttonBoxClicked)

    def layoutWidgets(self) -> None:
        layout_cal_form = QtWidgets.QFormLayout()
        layout_cal_form.addRow("Calibration Levels:", self.spinbox_levels)

        layout_table_form = QtWidgets.QFormLayout()
        layout_table_form.addRow("Units:", self.lineedit_units)
        layout_table_form.addRow("Weighting:", self.combo_weighting)
        layout_table_form.addRow("Averaging:", self.combo_averaging)

        layout_result_form = QtWidgets.QFormLayout()
        layout_result_form.addRow("RSQ:", self.lineedit_rsq)
        layout_result_form.addRow("Intercept:", self.lineedit_intercept)
        layout_result_form.addRow("Gradient:", self.lineedit_gradient)

        box_result = QtWidgets.QGroupBox("Result")
        box_result.setLayout(layout_result_form)

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addLayout(layout_cal_form)
        layout_left.addWidget(self.table)
        layout_left.addLayout(layout_table_form)
        layout_left.addWidget(box_result)

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

        layout_right = QtWidgets.QVBoxLayout()
        layout_right.addWidget(self.button_laser, 0, QtCore.Qt.AlignRight)
        layout_right.addWidget(self.canvas)
        layout_right.addLayout(layout_canvas_bar)

        # Main
        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_left, 1)
        layout_horz.addLayout(layout_right, 2)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_horz)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

    def accept(self) -> None:
        self.updateCalibration()
        self.applyPressed.emit(self)
        super().accept()

    def apply(self) -> None:
        self.updateCalibration()

    def draw(self) -> None:
        self.canvas.plot(
            self.dock.laser, self.combo_isotope.currentText(), self.viewconfig
        )
        self.canvas.plotLevels(self.spinbox_levels.value())
        self.canvas.draw()

    def updateCalibration(self) -> None:
        pass
        # self.dock.laser.calibration = self.calibration
        # self.dock.draw()

    def updateConcentrations(self) -> None:
        isotope = self.combo_isotope.currentText()
        if isotope in self.texts.keys():
            concentrations = self.texts[isotope]
            self.table.blockSignals(True)
            self.table.setColumnText(CalibrationTable.COLUMN_CONC, concentrations)
            self.table.blockSignals(False)

    def updateCounts(self) -> None:
        data = self.dock.laser.get(
            self.combo_isotope.currentText(), calibrated=False, extent=self.canvas.getView(),
        )
        if len(data) == 1:
            return

        if self.viewconfig["filtering"]["type"] != "None":
            filter_type, window, threshold = (
                self.viewconfig["filtering"][x] for x in ["type", "window", "threshold"]
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
        self.table.setColumnText(CalibrationTable.COLUMN_COUNT, text)
        self.table.blockSignals(False)

    def updateResults(self) -> None:
        # Clear results if not complete
        if not self.table.isComplete():
            self.lineedit_gradient.setText("")
            self.lineedit_intercept.setText("")
            self.lineedit_rsq.setText("")
            return

        x = np.array(
            self.table.columnText(CalibrationTable.COLUMN_CONC), dtype=np.float64
        )
        y = np.array(
            self.table.columnText(CalibrationTable.COLUMN_COUNT), dtype=np.float64
        )

        # Strip negative x values
        y = y[x >= 0.0]
        x = x[x >= 0.0]

        weighting = self.combo_weighting.currentText()
        if weighting == "x":
            weights = x
        elif weighting == "1/x":
            weights = 1.0 / x
        elif weighting == "1/(x^2)":
            weights = 1.0 / (x ** 2)
        else:  # Default is no weighting
            weights = None

        # Replace non finite values with one
        if weights is not None and not np.all(np.isfinite(weights)):
            self.lineedit_rsq.setText("Error")
            self.lineedit_intercept.setText("Invalid weighting")
            self.lineedit_gradient.setText("")
            return

        m, b, r2 = weighted_linreg(x, y, w=weights)
        self.lineedit_gradient.setText(f"{m:.4f}")
        self.lineedit_intercept.setText(f"{b:.4f}")
        self.lineedit_rsq.setText(f"{r2:.4f}")

        isotope = self.combo_isotope.currentText()
        self.calibration[isotope]["gradient"] = m
        self.calibration[isotope]["intercept"] = b

    @QtCore.pyqtSlot("QWidget*")
    def mouseSelectFinished(self, widget: QtWidgets.QWidget) -> None:
        if widget is not None and hasattr(widget, "laser"):
            self.dock = widget
            self.calibration = copy.deepcopy(widget.laser.calibration)
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
        self.show()
        self.draw()

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
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
            trim_left = self.dock.laser.convertRow(
                float(self.lineedit_left.text()),
                unit_from=self.combo_trim.currentText(),
                unit_to="um",
            )
        trim_right = self.dock.laser.extent()[1]
        if self.lineedit_right.text() != "":
            trim_right -= self.dock.laser.convertRow(
                float(self.lineedit_right.text()),
                unit_from=self.combo_trim.currentText(),
                unit_to="um",
            )
        self.canvas.setView(trim_left, trim_right, 0.0, self.canvas.extent[3])
        self.updateCounts()
        self.updateResults()

    def lineeditUnits(self) -> None:
        unit = self.lineedit_units.text()
        self.calibration[self.combo_isotope.currentText()]["unit"] = unit

    def spinBoxLevels(self) -> None:
        self.table.setRowCount(self.spinbox_levels.value())
        self.updateCounts()
        self.updateResults()
        self.draw()

    def tableItemChanged(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.text() != "":
            self.updateResults()
