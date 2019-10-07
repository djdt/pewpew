import copy
import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pew import Calibration

from pewpew.lib.numpyqt import NumpyArrayTableModel
from pewpew.lib.viewoptions import ViewOptions
from pewpew.validators import DoubleSignificantFiguresDelegate
from pewpew.widgets.canvases import BasicCanvas
from pewpew.widgets.dialogs import CalibrationCurveDialog
from pewpew.widgets.modelviews import BasicTableView
from pewpew.widgets.laser import LaserWidget

from .tool import ToolWidget

from typing import Any, List, Tuple


class StandardsTool(ToolWidget):
    def __init__(self, widget: LaserWidget):
        super().__init__(widget)
        self.setWindowTitle("Calibration Tool")

        self.calibration = copy.deepcopy(widget.laser.calibration)
        self.previous_isotope = ""
        current_isotope = self.widget.combo_isotopes.currentText()

        self.trim_left = 0
        self.trim_right = 0

        # Left side
        self.spinbox_levels = QtWidgets.QSpinBox()
        self.spinbox_levels.setMinimum(1)
        self.spinbox_levels.setMaximum(20)
        self.spinbox_levels.setValue(6)
        self.spinbox_levels.valueChanged.connect(self.spinBoxLevels)

        self.lineedit_units = QtWidgets.QLineEdit()
        self.lineedit_units.editingFinished.connect(self.lineeditUnits)
        self.lineedit_units.setText(self.calibration[current_isotope].unit)

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_weighting.addItems(["None", "x", "1/x", "1/(x^2)"])
        self.combo_weighting.setCurrentText(self.calibration[current_isotope].weighting)
        self.combo_weighting.currentIndexChanged.connect(self.comboWeighting)

        self.results_box = StandardsResultsBox()
        self.results_box.button.pressed.connect(self.showCurve)

        # Right side
        self.canvas = StandardsCanvas(self.viewspace.options, parent=self)

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
        self.combo_isotope.addItems(sorted(self.widget.laser.isotopes))
        self.combo_isotope.setCurrentText(current_isotope)
        self.combo_isotope.currentIndexChanged.connect(self.comboIsotope)

        self.table = StandardsTable(self.calibration[current_isotope], self)
        self.table.setRowCount(6)
        self.table.model().dataChanged.connect(self.completeChanged)
        self.table.model().dataChanged.connect(self.updateResults)

        self.button_apply = QtWidgets.QPushButton("Apply")
        self.button_apply.pressed.connect(self.apply)
        self.button_apply_all = QtWidgets.QPushButton("Apply to All")
        self.button_apply_all.pressed.connect(self.applyAll)

        self.layoutWidgets()
        self.refresh()

    def layoutWidgets(self) -> None:
        layout_cal_form = QtWidgets.QFormLayout()
        layout_cal_form.addRow("Calibration Levels:", self.spinbox_levels)

        layout_table_form = QtWidgets.QFormLayout()
        layout_table_form.addRow("Units:", self.lineedit_units)
        layout_table_form.addRow("Weighting:", self.combo_weighting)

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addLayout(layout_cal_form)
        layout_left.addWidget(self.table)
        layout_left.addLayout(layout_table_form)
        layout_left.addStretch(1)
        layout_left.addWidget(self.results_box)

        layout_box_trim = QtWidgets.QHBoxLayout()
        layout_box_trim.addWidget(QtWidgets.QLabel("Left:"))
        layout_box_trim.addWidget(self.lineedit_left)
        layout_box_trim.addWidget(QtWidgets.QLabel("Right:"))
        layout_box_trim.addWidget(self.lineedit_right)
        layout_box_trim.addWidget(self.combo_trim)

        box_trim = QtWidgets.QGroupBox("Trim")
        box_trim.setLayout(layout_box_trim)

        layout_canvas_bar = QtWidgets.QHBoxLayout()
        layout_canvas_bar.addWidget(box_trim, 0, QtCore.Qt.AlignLeft)
        layout_canvas_bar.addWidget(
            self.combo_isotope, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight
        )

        layout_right = QtWidgets.QVBoxLayout()
        layout_right.addWidget(self.canvas, 0, QtCore.Qt.AlignTop)
        layout_right.addStretch(1)
        layout_right.addLayout(layout_canvas_bar)

        self.layout_main.setDirection(QtWidgets.QBoxLayout.LeftToRight)
        self.layout_main.addLayout(layout_left, 0)
        self.layout_main.addLayout(layout_right, 1)

        self.layout_buttons.addStretch(1)
        self.layout_buttons.addWidget(self.button_apply, 0, QtCore.Qt.AlignRight)
        self.layout_buttons.addWidget(self.button_apply_all, 0, QtCore.Qt.AlignRight)

    def apply(self) -> None:
        self.widget.applyCalibration(self.calibration)

    def applyAll(self) -> None:
        self.viewspace.applyCalibration(self.calibration)

    def isComplete(self) -> bool:
        return self.table.isComplete()

    @QtCore.Slot()
    def completeChanged(self) -> None:
        enabled = self.isComplete()
        self.button_apply.setEnabled(enabled)
        self.button_apply_all.setEnabled(enabled)

    def refresh(self) -> None:
        isotope = self.combo_isotope.currentText()
        if isotope not in self.widget.laser.isotopes:
            return

        data = self.widget.laser.get(isotope, calibrate=False, flat=True)
        data = data[:, self.trim_left : data.shape[1] - self.trim_right]
        if data.size == 0:
            return

        extent = self.widget.laser.config.data_extent(data.shape)
        self.canvas.drawData(data, extent)
        self.canvas.drawLevels(StandardsTable.ROW_LABELS, self.spinbox_levels.value())
        self.canvas.draw()

        buckets = np.array_split(data, self.spinbox_levels.value(), axis=0)
        self.table.setCounts([np.nanmean(b) for b in buckets])

    def updateResults(self) -> None:
        # Clear results if not complete
        if not self.isComplete():
            self.results_box.clear()
            return
        else:
            isotope = self.combo_isotope.currentText()
            self.results_box.update(self.calibration[isotope])

    def widgetChanged(self) -> None:
        self.calibration = copy.deepcopy(self.widget.laser.calibration)
        # Prevent currentIndexChanged being emmited
        self.combo_isotope.blockSignals(True)
        self.combo_isotope.clear()
        self.combo_isotope.addItems(sorted(self.widget.laser.isotopes))
        self.combo_isotope.setCurrentText(self.widget.combo_isotopes.currentText())
        self.combo_isotope.blockSignals(False)

        self.lineedit_left.setText("")
        self.lineedit_right.setText("")

        isotope = self.combo_isotope.currentText()
        self.combo_weighting.setCurrentText(self.calibration[isotope].weighting)
        self.lineedit_units.setText(self.calibration[isotope].unit)
        self.table.model().setCalibration(self.calibration[isotope])

        self.refresh()
        self.updateResults()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.MouseButtonDblClick and isinstance(
            obj.parent(), LaserWidget
        ):
            self.widget = obj.parent()
            self.widgetChanged()
            self.endMouseSelect()
        return False

    # Widget callbacks
    def comboTrim(self, text: str) -> None:
        if self.combo_trim.currentText() == "rows":
            self.lineedit_left.setValidator(QtGui.QIntValidator(0, 1e9))
            self.lineedit_right.setValidator(QtGui.QIntValidator(0, 1e9))
        else:
            self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
            self.lineedit_right.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_left.setText("")
        self.lineedit_right.setText("")
        self.trim_left = 0
        self.trim_right = 0

    def comboIsotope(self, text: str) -> None:
        isotope = self.combo_isotope.currentText()
        self.table.model().setCalibration(self.calibration[isotope])

        if self.calibration[isotope].unit != "":
            self.lineedit_units.setText(self.calibration[isotope].unit)
        else:
            self.calibration[isotope].unit = self.lineedit_units.text()
        if self.calibration[isotope].weighting is not None:
            self.combo_weighting.setCurrentText(self.calibration[isotope].weighting)
        else:
            self.calibration[isotope].weighting = self.combo_weighting.currentText()

        self.refresh()

    def comboWeighting(self, index: int) -> None:
        isotope = self.combo_isotope.currentText()
        self.calibration[isotope].weighting = self.combo_weighting.currentText()
        self.calibration[isotope].update_linreg()
        self.updateResults()

    def lineEditTrim(self) -> None:
        if self.lineedit_left.text() == "":
            trim_left = 0.0
        else:
            trim_left = float(self.lineedit_left.text())
        if self.lineedit_right.text() == "":
            trim_right = 0.0
        else:
            trim_right = float(self.lineedit_right.text())

        # Convert units
        units = self.combo_trim.currentText()
        multiplier = 1.0
        if units == "μm":
            multiplier /= self.widget.laser.config.get_pixel_width()
        if units == "s":
            multiplier /= self.widget.laser.config.scantime

        self.trim_left = int(trim_left * multiplier)
        self.trim_right = int(trim_right * multiplier)

        self.refresh()

    def lineeditUnits(self) -> None:
        isotope = self.combo_isotope.currentText()
        unit = self.lineedit_units.text()
        self.calibration[isotope].unit = unit

    def showCurve(self) -> QtWidgets.QDialog:
        dlg = CalibrationCurveDialog(
            self.calibration[self.combo_isotope.currentText()], parent=self
        )
        dlg.show()
        return dlg

    def spinBoxLevels(self) -> None:
        self.table.setRowCount(self.spinbox_levels.value())
        self.refresh()


class StandardsCanvas(BasicCanvas):
    def __init__(self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)
        self.viewoptions = viewoptions
        # Restore view limits
        self.ax = self.figure.subplots()
        self.ax.set_facecolor("black")
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        div = make_axes_locatable(self.ax)
        self.bax = div.append_axes("left", size=0.2, pad=0, sharey=self.ax)
        self.bax.set_facecolor("black")
        self.bax.get_xaxis().set_visible(False)
        self.bax.get_yaxis().set_visible(False)

        self.image: AxesImage = None

    def drawData(
        self, data: np.ndarray, extent: Tuple[float, float, float, float]
    ) -> None:
        self.ax.clear()
        self.image = self.ax.imshow(
            data,
            extent=extent,
            cmap=self.viewoptions.image.cmap,
            interpolation=self.viewoptions.image.interpolation,
            alpha=self.viewoptions.image.alpha,
            aspect="equal",
            origin="upper",
        )

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
        text = text.rstrip("\n")
        data += "</table>"

        mime = QtCore.QMimeData()
        mime.setHtml(data)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def clear(self) -> None:
        for le in self.lineedits:
            le.setText("")
        self.button.setEnabled(False)

    def update(self, calibration: Calibration) -> None:
        for v, le in zip(
            [calibration.rsq, calibration.gradient, calibration.intercept],
            self.lineedits,
        ):
            le.setText(f"{v:.4f}")
        self.button.setEnabled(True)


class CalibrationPointsTableModel(NumpyArrayTableModel):
    def __init__(self, calibration: Calibration, parent: QtCore.QObject = None):
        self.calibration = calibration
        if self.calibration.points is None or self.calibration.points.size == 0:
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

    def setCalibration(self, calibration: Calibration) -> None:
        self.beginResetModel()
        self.calibration = calibration
        new_array = np.full_like(self.array, np.nan)
        if self.calibration.points is not None and self.calibration.points.size > 0:
            min_row = np.min((new_array.shape[0], self.calibration.points.shape[0]))
            new_array[:min_row] = self.calibration.points[:min_row]
        self.array = new_array
        self.endResetModel()

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> str:
        value = super().data(index, role)
        if value == "nan":
            return ""
        return value

    def setData(
        self, index: QtCore.QModelIndex, value: Any, role: int = QtCore.Qt.EditRole
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
        self, position: int, columns: int, parent: QtCore.QModelIndex = None
    ) -> bool:
        return False

    def removeColumns(
        self, position: int, columns: int, parent: QtCore.QModelIndex = None
    ) -> bool:
        return False

    def updateCalibration(self, *args) -> None:
        if np.count_nonzero(np.nan_to_num(self.array[:, 0])) < 2:
            self.calibration._points = None
        else:
            self.calibration.points = self.array

        self.calibration.update_linreg()


class StandardsTable(BasicTableView):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]
    COLUMN_CONC = 0
    COLUMN_COUNT = 1

    def __init__(self, calibration: Calibration, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow
        )
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
                np.nan_to_num(self.model().array[:, StandardsTable.COLUMN_CONC])
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
