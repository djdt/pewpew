import copy
import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.backend_bases import LocationEvent, PickEvent, MouseEvent
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pew import Calibration
from pew.calc import get_weights

from pewpew.lib.numpyqt import NumpyArrayTableModel
from pewpew.lib.viewoptions import ViewOptions
from pewpew.validators import DoubleSignificantFiguresDelegate
from pewpew.widgets.canvases import InteractiveCanvas
from pewpew.widgets.dialogs import CalibrationCurveDialog
from pewpew.widgets.modelviews import BasicTableView
from pewpew.widgets.laser import LaserWidget

from .tool import ToolWidget

from typing import Any, Dict, List, Tuple


class StandardsTool(ToolWidget):
    WEIGHTINGS = ["None", "1/σ²", "x", "1/x", "1/x²"]

    def __init__(self, widget: LaserWidget):
        super().__init__(widget)
        self.setWindowTitle("Calibration Tool")

        self.calibration: Dict[str, Calibration] = None
        self.previous_isotope = ""
        self.standard_deviations: np.ndarray = None

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

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_weighting.addItems(self.WEIGHTINGS)
        self.combo_weighting.currentIndexChanged.connect(self.comboWeighting)

        self.results_box = StandardsResultsBox()
        self.results_box.button.pressed.connect(self.showCurve)

        # Right side
        self.canvas = StandardsCanvas(self.viewspace.options, parent=self)

        self.lineedit_left = QtWidgets.QLineEdit("0")
        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_left.editingFinished.connect(self.lineEditTrim)
        self.lineedit_right = QtWidgets.QLineEdit("0")
        self.lineedit_right.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_right.editingFinished.connect(self.lineEditTrim)

        self.combo_trim = QtWidgets.QComboBox()
        self.combo_trim.addItems(["row", "s", "μm"])
        self.combo_trim.setCurrentIndex(1)
        self.combo_trim.currentTextChanged.connect(self.comboTrim)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.comboIsotope)

        self.table = StandardsTable(Calibration(), self)
        self.table.setRowCount(6)
        self.table.model().dataChanged.connect(self.completeChanged)
        self.table.model().dataChanged.connect(self.updateResults)

        self.button_apply = QtWidgets.QPushButton("Apply")
        self.button_apply.pressed.connect(self.apply)
        self.button_apply_all = QtWidgets.QPushButton("Apply to All")
        self.button_apply_all.pressed.connect(self.applyAll)

        self.layoutWidgets()
        self.widgetChanged()

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
        if len(self.canvas.v_guides) == 0:
            self.canvas.drawVerticalGuides()
        self.canvas.drawLevels(StandardsTable.ROW_LABELS, self.spinbox_levels.value())
        self.canvas.draw()
        self.canvas.update_background(None)
        self.canvas.blitGuides()

        buckets = np.array_split(data, self.spinbox_levels.value(), axis=0)
        self.table.setCounts([np.nanmean(b) for b in buckets])

    def updateWeights(self) -> None:
        isotope = self.combo_isotope.currentText()
        weighting = self.combo_weighting.currentText()
        if weighting == "1/σ²":
            data = self.widget.laser.get(isotope, calibrate=False, flat=True)
            data = data[:, self.trim_left : data.shape[1] - self.trim_right]
            buckets = np.array_split(data, self.spinbox_levels.value(), axis=0)
            weights = 1 / np.square(np.array([np.nanstd(b) for b in buckets]))
            self.calibration[isotope].weights = weights
        else:
            self.calibration[isotope].weights = weighting

    def updateResults(self) -> None:
        # Make sure weights are up to date
        self.updateWeights()
        # Clear results if not complete
        if not self.isComplete():
            self.results_box.clear()
            return
        else:
            isotope = self.combo_isotope.currentText()
            self.calibration[isotope].update_linreg()
            self.results_box.update(self.calibration[isotope])

    def widgetChanged(self) -> None:
        self.label_current.setText(self.widget.laser.name)

        self.calibration = copy.deepcopy(self.widget.laser.calibration)
        # Prevent currentIndexChanged being emmited
        self.combo_isotope.blockSignals(True)
        self.combo_isotope.clear()
        self.combo_isotope.addItems(sorted(self.widget.laser.isotopes))
        self.combo_isotope.setCurrentText(self.widget.combo_isotopes.currentText())
        self.combo_isotope.blockSignals(False)

        self.lineedit_left.setText("0")
        self.lineedit_right.setText("0")

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
            self.endMouseSelect()
        return False

    # Widget callbacks
    def comboTrim(self, text: str) -> None:
        if text == "row":
            self.lineedit_left.setValidator(QtGui.QIntValidator(0, 1e9))
            self.lineedit_right.setValidator(QtGui.QIntValidator(0, 1e9))
        else:
            self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
            self.lineedit_right.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_left.setText("0")
        self.lineedit_right.setText("0")
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

    def getWeights(self, weighting: str) -> np.ndarray:
        isotope = self.combo_isotope.currentText()
        if weighting == "1/σ²":
            return 1 / np.power(self.standard_deviations, 2)
        else:
            return get_weights(self.calibration[isotope].concentrations(), weighting)

    def comboWeighting(self, index: int) -> None:
        isotope = self.combo_isotope.currentText()
        self.calibration[isotope].weighting = self.combo_weighting.currentText()
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


class StandardsCanvas(InteractiveCanvas):
    def __init__(self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None):
        super().__init__(parent=parent)
        self.viewoptions = viewoptions
        self.button = 1

        self.connect_event("draw_event", self.update_background)
        self.connect_event("resize_event", self._resize)

        self.guides_need_draw = True

        self.image: AxesImage = None
        self.background = None
        self.h_guides: List[Line2D] = []
        self.v_guides: List[Line2D] = []

        self.redrawFigure()

    def ignore_event(self, event: LocationEvent) -> bool:
        if event.name not in [
            "pick_event",
            "button_press_event",
            "button_release_event",
            "motion_notify_event",
        ]:
            return True
        elif (
            event.name in ["button_press_event", "button_release_event"]
            and event.button != self.button
        ):
            return True

        if event.inaxes != self.ax:
            return True

        return super().ignore_event(event)

    def _resize(self, event) -> None:
        self.guides_need_draw = True

    def update_background(self, event) -> None:
        self.background = self.copy_from_bbox(self.ax.bbox)
        if self.guides_need_draw:
            self.blitGuides()

    def onpick(self, event: PickEvent) -> None:
        pass

    def move(self, event: MouseEvent) -> None:
        if self.picked_artist is None:
            return

        if self.picked_artist in self.v_guides:
            x, y = self.ax.transAxes.inverted().transform([event.x, event.y])
            self.picked_artist.set_xdata([x, x])
            self.blitGuides()

    def press(self, event: MouseEvent) -> None:
        pass

    def release(self, event: MouseEvent) -> None:
        if self.picked_artist in self.v_guides:
            x, _x = self.picked_artist.get_xdata()
            shape = self.image.get_array().shape
            px = 1.0 / shape[1]  # Axes coords
            # Snap guide
            x = x - (x % px)
            self.picked_artist.set_xdata([x, x])
            self.blitGuides()
        self.picked_artist = None

    def redrawFigure(self) -> None:
        self.background = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(facecolor="black")
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        div = make_axes_locatable(self.ax)
        self.bax = div.append_axes("left", size=0.2, pad=0, sharey=self.ax)
        self.bax.set_facecolor("black")
        self.bax.get_xaxis().set_visible(False)
        self.bax.get_yaxis().set_visible(False)

    def drawData(
        self, data: np.ndarray, extent: Tuple[float, float, float, float]
    ) -> None:
        if self.image is not None:
            self.image.remove()
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

        # Draw lines
        for line in self.h_guides:
            line.remove()
        self.h_guides = []

        for frac in np.linspace(1.0 - ax_fraction, ax_fraction, levels - 1):
            line = Line2D(
                (0.0, 1.0),
                (frac, frac),
                transform=self.ax.transAxes,
                color="white",
                linestyle="--",
                path_effects=[withStroke(linewidth=2.0, foreground="black")],
                linewidth=1.0,
            )
            self.h_guides.append(line)
            self.ax.add_artist(line)

    def drawVerticalGuides(self, ax_pos: Tuple[float, float] = (0.1, 0.9)) -> None:
        self.v_guides = []

        for pos in ax_pos:
            line = Line2D(
                (pos, pos),
                (0.0, 1.0),
                transform=self.ax.transAxes,
                color="white",
                linestyle="-",
                path_effects=[withStroke(linewidth=2.0, foreground="black")],
                linewidth=1.0,
                picker=20,
                animated=True,
            )
            self.v_guides.append(line)
            self.ax.add_artist(line)

    def blitGuides(self) -> None:
        if self.background is not None:
            self.restore_region(self.background)

        for line in self.v_guides:
            self.ax.draw_artist(line)

        self.update()
        self.guides_need_draw = False


class StandardsResultsBox(QtWidgets.QGroupBox):
    LABELS = ["RSQ", "Gradient", "Intercept", "Y-error"]

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
            [
                calibration.rsq,
                calibration.gradient,
                calibration.intercept,
                calibration.yerr,
            ],
            self.lineedits,
        ):
            le.setText(f"{v:.4f}" if v is not None else "")
        self.button.setEnabled(True)


class CalibrationPointsTableModel(NumpyArrayTableModel):
    def __init__(self, calibration: Calibration, parent: QtCore.QObject = None):
        self.calibration = calibration
        if self.calibration.points is None or self.calibration.points.size == 0:
            array = np.array([[np.nan, np.nan]], dtype=np.float64)
        else:
            array = self.calibration.points
        super().__init__(array, parent)

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
            new_array[:min_row, :2] = self.calibration.points[:min_row]
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
            self.calibration.points = self.array[:, :2]


class StandardsTable(BasicTableView):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]
    COLUMN_CONC = 0
    COLUMN_COUNT = 1
    COLUMN_WEIGHTS = 2

    def __init__(self, calibration: Calibration, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow
        )
        model = CalibrationPointsTableModel(calibration, self)
        self.setModel(model)
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
