import copy
import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from laserlib import LaserCalibration

from pewpew.lib.numpyqt import NumpyArrayTableModel
from pewpew.validators import DoubleSignificantFiguresDelegate
from pewpew.widgets.canvases import LaserCanvas
from pewpew.widgets.docks import LaserImageDock
from pewpew.widgets.dialogs import CalibrationCurveDialog
from pewpew.widgets.views import BasicTableView

from .tool import Tool

from typing import Any, List


class StandardsTool(Tool):
    def __init__(
        self,
        dock: LaserImageDock,
        viewconfig: dict,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration Standards Tool")

        self.dock = dock
        self.dockarea = dock.parent()
        self.previous_isotope = ""
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
        self.combo_isotope.addItems(sorted(self.dock.laser.isotopes))
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
            self.combo_isotope.addItems(sorted(self.dock.laser.isotopes))
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
        self.updateCounts()

    def lineeditUnits(self) -> None:
        isotope = self.combo_isotope.currentText()
        unit = self.lineedit_units.text()
        self.calibrations[isotope].unit = unit

    def spinBoxLevels(self) -> None:
        self.table.setRowCount(self.spinbox_levels.value())
        self.updateCounts()
        self.draw()


class StandardsCanvas(LaserCanvas):
    def __init__(self, viewconfig: dict, parent: QtWidgets.QWidget = None):
        options = {"colorbar": False, "scalebar": False, "label": False}
        super().__init__(viewconfig, options=options, parent=parent)
        div = make_axes_locatable(self.ax)
        self.bax = div.append_axes("left", size=0.2, pad=0, sharey=self.ax)
        self.bax.get_xaxis().set_visible(False)
        self.bax.get_yaxis().set_visible(False)
        self.bax.set_facecolor("black")

    def drawLevels(self, texts: List[str], levels: int) -> None:
        self.bax.clear()
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

        for i, frac in enumerate(np.linspace(1.0, ax_fraction, levels)):
            text = Text(
                x=0.5,
                y=frac - (ax_fraction / 2.0),
                text=texts[i],
                transform=self.bax.transAxes,
                color="white",
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="center",
            )
            self.bax.add_artist(text)


class StandardsResultsBox(QtWidgets.QGroupBox):
    LABELS = ["RSQ", "Gradient", "Intercept"]

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Results", parent)
        self.lineedits: List[QtWidgets.QLineEdit] = []
        self.button = QtWidgets.QPushButton("Plot")
        self.button.setEnabled(False)

        layout = QtWidgets.QFormLayout()

        for label in StandardsResultsBox.LABELS:
            le = QtWidgets.QLineEdit()
            le.setReadOnly(True)

            layout.addRow(label, le)
            self.lineedits.append(le)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button, 0, QtCore.Qt.AlignRight)
        layout.addRow(button_layout)
        self.setLayout(layout)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        copy_action = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("edit-copy"), "Copy All", self
        )
        copy_action.triggered.connect(self.copy)

        menu.addAction(copy_action)

        menu.popup(event.globalPos())

    def copy(self) -> None:
        data = (
            '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
            "<table>"
        )
        text = ""

        for label, lineedit in zip(StandardsResultsBox.LABELS, self.lineedits):
            value = lineedit.text()
            data += f"<tr><td>{label}</td><td>{value}</td></tr>"
            text += f"{label}\t{value}\n"
        data += "</table>"

        mime = QtCore.QMimeData()
        mime.setHtml(data)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def clear(self) -> None:
        for le in self.lineedits:
            le.setText("")
        self.button.setEnabled(False)

    def update(self, calibration: LaserCalibration) -> None:
        for v, le in zip(
            [calibration.rsq, calibration.gradient, calibration.intercept],
            self.lineedits,
        ):
            le.setText(f"{v:.4f}")
        self.button.setEnabled(True)


class CalibrationPointsTableModel(NumpyArrayTableModel):
    def __init__(self, calibration: LaserCalibration, parent: QtCore.QObject = None):
        self.calibration = calibration
        if self.calibration.points.size == 0:
            points = np.array([[np.nan, np.nan]], dtype=np.float64)
        else:
            points = self.calibration.points
        super().__init__(points, parent)

        self.alphabet_rows = True
        self.fill_value = np.nan

        self.dataChanged.connect(self.updateCalibration)
        self.rowsInserted.connect(self.updateCalibration)
        self.rowsRemoved.connect(self.updateCalibration)
        self.modelReset.connect(self.updateCalibration)

    def setCalibration(self, calibration: LaserCalibration) -> None:
        self.beginResetModel()
        self.calibration = calibration
        new_array = np.full_like(self.array, np.nan)
        if self.calibration.points.size > 0:
            min_row = np.min((new_array.shape[0], self.calibration.points.shape[0]))
            new_array[:min_row] = self.calibration.points[:min_row]
        self.array = new_array
        self.endResetModel()

    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole
    ) -> str:
        value = super().data(index, role)
        if value == "nan":
            return ""
        return value

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: Any,
        role: int = QtCore.Qt.EditRole,
    ) -> bool:
        return super().setData(index, np.nan if value == "" else value, role)

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled

        flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if index.column() == 0:
            flags = QtCore.Qt.ItemIsEditable | flags
        return flags

    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int
    ) -> str:
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Horizontal:
            return ("Concentration", "Counts")[section]
        else:
            if self.alphabet_rows:
                return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[section]
            return str(section)

    def insertColumns(
        self,
        position: int,
        columns: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        return False

    def removeColumns(
        self,
        position: int,
        columns: int,
        parent: QtCore.QModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        return False

    def updateCalibration(self, *args) -> None:
        if np.count_nonzero(~np.isnan(self.array[:, 0])) == 0:
            self.calibration.points = np.array([], dtype=np.float64)
        else:
            self.calibration.points = self.array

        self.calibration.update_linreg()


class StandardsTable(BasicTableView):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]
    COLUMN_CONC = 0
    COLUMN_COUNT = 1

    def __init__(self, calibration, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        model = CalibrationPointsTableModel(calibration, self)
        self.setModel(model)
        # self.setHorizontalHeader(["Concentration", "Counts"])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.setItemDelegate(DoubleSignificantFiguresDelegate(4))

    def isComplete(self) -> bool:
        if np.nan in self.model().array[:, StandardsTable.COLUMN_COUNT]:
            return False
        if (
            np.count_nonzero(
                ~np.isnan(self.model().array[:, StandardsTable.COLUMN_CONC])
            )
            < 2
        ):
            return False
        return True

    def setCounts(self, counts: np.ndarray) -> None:
        self.model().blockSignals(True)
        for i in range(0, self.model().rowCount()):
            self.model().setData(
                self.model().index(i, StandardsTable.COLUMN_COUNT), counts[i]
            )
        self.model().blockSignals(False)
        self.model().dataChanged.emit(
            QtCore.QModelIndex(), QtCore.QModelIndex(), [QtCore.Qt.EditRole]
        )

    def setRowCount(self, rows: int) -> None:
        current_rows = self.model().rowCount()
        if current_rows < rows:
            self.model().insertRows(current_rows, rows - current_rows)
        elif current_rows > rows:
            self.model().removeRows(rows, current_rows - rows)
