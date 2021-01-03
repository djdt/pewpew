import copy
import numpy as np

from PySide2 import QtCore, QtWidgets

# from matplotlib.artist import Artist
# from matplotlib.backend_bases import PickEvent, MouseEvent
# from matplotlib.lines import Line2D
# from matplotlib.patheffects import withStroke
# from matplotlib.transforms import blended_transform_factory

from pewlib import Calibration

from pewpew.graphics.overlaygraphics import OverlayView
from pewpew.graphics.options import GraphicsOptions

from pewpew.lib.numpyqt import NumpyArrayTableModel
# from pewpew.lib.viewoptions import ViewOptions
# from pewpew.lib.mpltools import LabeledLine2D
from pewpew.validators import DoubleSignificantFiguresDelegate
# from pewpew.widgets.canvases import InteractiveImageCanvas
from pewpew.widgets.dialogs import CalibrationCurveDialog
from pewpew.widgets.modelviews import BasicTable, BasicTableView
from pewpew.widgets.laser import LaserWidget

from .tool import ToolWidget

from typing import Any, Dict, List, Tuple


class StandardsTool(ToolWidget):
    WEIGHTINGS = ["Equal", "1/σ²", "x", "1/x", "1/x²"]

    def __init__(self, widget: LaserWidget):
        super().__init__(widget, control_label="Calibration", apply_all=True)
        self.setWindowTitle("Calibration Tool")

        self.calibration: Dict[str, Calibration] = None
        self.previous_isotope = ""

        # Left side
        self.spinbox_levels = QtWidgets.QSpinBox()
        self.spinbox_levels.setMinimum(1)
        self.spinbox_levels.setMaximum(20)
        self.spinbox_levels.setValue(6)
        self.spinbox_levels.valueChanged.connect(self.spinBoxLevels)

        self.lineedit_units = QtWidgets.QLineEdit()
        self.lineedit_units.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.lineedit_units.editingFinished.connect(self.lineeditUnits)

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_weighting.addItems(self.WEIGHTINGS)
        self.combo_weighting.currentIndexChanged.connect(self.comboWeighting)

        self.results = StandardsResultsTable(self)
        self.results.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow
        )
        self.results.horizontalHeader().setStretchLastSection(True)
        self.button_plot = QtWidgets.QPushButton("Plot")
        self.button_plot.pressed.connect(self.showCurve)

        # Right side
        self.graphics = StandardsCanvas(self.viewspace.options, parent=self)
        self.graphics.state.discard("move")  # Prevent moving
        self.graphics.state.discard("scroll")  # Prevent scroll zoom
        self.graphics.guidesChanged.connect(self.updateCounts)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.comboIsotope)

        self.table = StandardsTable(Calibration(), self)
        self.table.setRowCount(6)
        self.table.model().dataChanged.connect(self.completeChanged)
        self.table.model().dataChanged.connect(self.updateResults)

        # Initialise
        self.calibration = copy.deepcopy(self.widget.laser.calibration)
        # Prevent currentIndexChanged being emmited
        self.combo_isotope.blockSignals(True)
        self.combo_isotope.clear()
        self.combo_isotope.addItems(self.widget.laser.isotopes)
        self.combo_isotope.setCurrentText(self.widget.combo_isotope.currentText())
        self.combo_isotope.blockSignals(False)

        isotope = self.combo_isotope.currentText()
        self.combo_weighting.setCurrentText(self.calibration[isotope].weighting)
        self.lineedit_units.setText(self.calibration[isotope].unit)
        self.table.model().setCalibration(self.calibration[isotope])

        layout_cal_form = QtWidgets.QFormLayout()
        layout_cal_form.addRow("Levels:", self.spinbox_levels)

        layout_table_form = QtWidgets.QFormLayout()
        layout_table_form.addRow("Units:", self.lineedit_units)
        layout_table_form.addRow("Weighting:", self.combo_weighting)

        box_results = QtWidgets.QGroupBox("Results")
        layout_results = QtWidgets.QHBoxLayout()
        layout_results.addWidget(self.results, 0)
        layout_results.addWidget(
            self.button_plot, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom
        )
        layout_results.addStretch(1)
        box_results.setLayout(layout_results)

        layout_graphics = QtWidgets.QVBoxLayout()
        layout_graphics.addWidget(self.graphics)
        self.box_graphics.setLayout(layout_graphics)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addLayout(layout_cal_form)
        layout_controls.addWidget(self.table)
        layout_controls.addLayout(layout_table_form)
        self.box_controls.setLayout(layout_controls)

        self.layout_bottom.addWidget(box_results, 0, QtCore.Qt.AlignLeft)
        self.layout_bottom.addWidget(
            self.combo_isotope, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight
        )

        self.refresh()
        self.updateResults()

    def apply(self) -> None:
        self.widget.applyCalibration(self.calibration)  # pragma: no cover

    def applyAll(self) -> None:
        self.viewspace.applyCalibration(self.calibration)  # pragma: no cover

    def isComplete(self) -> bool:
        return self.table.isComplete()

    def refresh(self) -> None:
        isotope = self.combo_isotope.currentText()
        if isotope not in self.widget.laser.isotopes:  # pragma: no cover
            return

        data = self.widget.laser.get(isotope, calibrate=False, flat=True)
        extent = self.widget.laser.config.data_extent(data.shape)
        self.graphics.drawData(data, extent)
        # Update view limits
        self.canvas.view_limits = self.canvas.extentForAspect(extent)
        # Redraw guides if number of levels change
        if len(self.canvas.level_guides) - 1 != self.spinbox_levels.value():
            self.canvas.drawLevels(
                StandardsTable.ROW_LABELS, self.spinbox_levels.value()
            )
        # Draw vert guides after so they get pick priority
        if len(self.canvas.edge_guides) == 0:
            w = extent[1] - extent[0]
            px = w / data.shape[1]
            x0, x1 = px * np.round(np.array([0.05, 0.95]) * w / px)
            self.canvas.drawEdgeGuides((x0, x1))

        self.canvas.guides_need_draw = True
        self.canvas.draw()

        self.updateCounts()

    def updateWeights(self) -> None:
        isotope = self.combo_isotope.currentText()
        wstr = self.combo_weighting.currentText()
        if wstr == "1/σ²":
            if self.calibration[isotope].x.size > 0:
                data = self.graphics.image.get_array()
                trim_left, trim_right = self.graphics.getCurrentTrim()
                data = data[:, trim_left:trim_right]
                levels = self.graphics.getCurrentLevels()
                order = np.argsort(levels)
                buckets = np.split(data, levels[order], axis=0)
                weights = 1.0 / np.square(
                    np.array([np.nanstd(buckets[i + 1]) for i in order])
                )
            else:
                weights = np.empty(0, dtype=np.float64)
            self.calibration[isotope].weights = (wstr, weights)
        else:
            if wstr == "1/x²":
                wstr = "1/(x^2)"
            self.calibration[isotope].weights = wstr

    def updateCounts(self) -> None:
        data = self.graphics.image.get_array()
        trim_left, trim_right = self.graphics.getCurrentTrim()
        data = data[:, trim_left:trim_right]
        if data.size == 0:  # pragma: no cover
            return

        levels = self.graphics.getCurrentLevels()
        order = np.argsort(levels)
        buckets = np.split(data, levels[order], axis=0)
        self.table.setCounts([np.nanmean(buckets[i + 1]) for i in order])

    def updateResults(self) -> None:
        # Make sure weights are up to date
        self.updateWeights()
        # Clear results if not complete
        if not self.isComplete():
            self.results.clearResults()
            self.button_plot.setEnabled(False)
            return
        else:
            isotope = self.combo_isotope.currentText()
            self.calibration[isotope].update_linreg()
            self.results.updateResults(self.calibration[isotope])
            self.button_plot.setEnabled(True)

    # Widget callbacks
    def comboIsotope(self, text: str) -> None:
        isotope = self.combo_isotope.currentText()
        self.table.model().setCalibration(self.calibration[isotope])

        self.lineedit_units.setText(self.calibration[isotope].unit)

        if self.calibration[isotope].weighting is not None:
            self.combo_weighting.setCurrentText(self.calibration[isotope].weighting)
        else:  # pragma: no cover
            self.calibration[isotope].weighting = "Equal"

        self.refresh()

    def comboWeighting(self, index: int) -> None:
        isotope = self.combo_isotope.currentText()
        self.calibration[isotope].weighting = self.combo_weighting.currentText()
        self.updateResults()

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
        self.table.updateGeometry()
        self.refresh()


class StandardsCanvas(InteractiveImageCanvas):
    guidesChanged = QtCore.Signal()

    def __init__(self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None):
        super().__init__(widget_button=1, state=(), parent=parent)
        self.viewoptions = viewoptions

        self.connect_event("draw_event", self.update_background)
        self.connect_event("resize_event", self._resize)
        self.connect_event("pick_event", self._pick)

        self.guides_need_draw = True

        self.background = None
        self.level_guides: List[LabeledLine2D] = []
        self.edge_guides: List[Line2D] = []

        self.picked_artist: Artist = None

        self.drawFigure()

    def _resize(self, event) -> None:
        self.guides_need_draw = True

    def _pick(self, event: PickEvent) -> None:
        if self.ignore_event(event.mouseevent):
            return
        if event.mouseevent.button == self.widget_button:
            self.picked_artist = event.artist
        else:
            self.picked_artist = None

    def update_background(self, event) -> None:
        self.background = self.copy_from_bbox(self.ax.bbox)
        if self.guides_need_draw:
            self.blitGuides()

    def move(self, event: MouseEvent) -> None:
        if self.picked_artist is not None:
            if self.picked_artist in self.level_guides:
                self.picked_artist.set_ydata([event.ydata, event.ydata])
            elif self.picked_artist in self.edge_guides:
                self.picked_artist.set_xdata([event.xdata, event.xdata])
            self.blitGuides()

    def release(self, event: MouseEvent) -> None:
        if self.picked_artist is not None:
            x = self.picked_artist.get_xdata()[0]
            y = self.picked_artist.get_ydata()[0]
            shape = self.image.get_array().shape
            x0, x1, y0, y1 = self.extent
            # Snap to pixels
            pxy = (x1 - x0, y1 - y0) / np.array((shape[1], shape[0]))
            x, y = pxy * np.round((x, y) / pxy)

            if self.picked_artist in self.level_guides:
                self.picked_artist.set_ydata([y, y])
            elif self.picked_artist in self.edge_guides:
                self.picked_artist.set_xdata([x, x])

            self.blitGuides()
            self.guidesChanged.emit()
            self.picked_artist = None

    # def drawFigure(self) -> None:
    #     self.background = None
    #     self.figure.clear()
    #     self.ax = self.figure.add_subplot(facecolor="black")
    #     self.ax.get_xaxis().set_visible(False)
    #     self.ax.get_yaxis().set_visible(False)

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
            origin="lower",
        )

    def drawLevels(self, texts: List[str], levels: int) -> None:
        x0, x1, y0, y1 = self.extent
        pos = np.linspace(y0, y1, num=levels + 1)
        # Snap
        py = (y1 - y0) / self.image.get_array().shape[0]
        pos = py * np.round(pos / py)
        self.drawLevelGuides(pos, texts)
        # First guide is just for label
        self.level_guides[0].set_picker(False)
        self.level_guides[-1].set_picker(False)
        self.level_guides[-1].text.set_text("")
        # self.level_guides[0].set_visible()

    def drawLevelGuides(self, ypos: List[float], texts: List[str]) -> None:
        for line in self.level_guides:
            line.remove()
        self.level_guides = []

        textprops = dict(
            color="white",
            fontsize=12,
            path_effects=[withStroke(linewidth=1.5, foreground="black")],
            horizontalalignment="left",
            verticalalignment="top",
        )

        for y, text in zip(ypos, texts):
            line = LabeledLine2D(
                (0.0, 1.0),
                (y, y),
                transform=blended_transform_factory(
                    self.ax.transAxes, self.ax.transData
                ),
                color="white",
                linestyle="--",
                path_effects=[withStroke(linewidth=2.0, foreground="black")],
                linewidth=1.0,
                picker=True,
                pickradius=5,
                animated=True,
                label=text,
                label_offset=(5, -5),
                textprops=textprops,
            )
            self.level_guides.append(line)
            self.ax.add_artist(line)

    def drawEdgeGuides(self, xpos: Tuple[float, float]) -> None:
        for line in self.edge_guides:
            line.remove()
        self.edge_guides = []

        for x in xpos:
            line = Line2D(
                (x, x),
                (0.0, 1.0),
                transform=blended_transform_factory(
                    self.ax.transData, self.ax.transAxes
                ),
                color="white",
                linestyle="-",
                path_effects=[withStroke(linewidth=2.0, foreground="black")],
                linewidth=1.0,
                picker=True,
                pickradius=5,
                animated=True,
            )
            self.edge_guides.append(line)
            self.ax.add_artist(line)

    def blitGuides(self) -> None:
        if self.background is not None:
            self.restore_region(self.background)

        for a in self.level_guides:
            self.ax.draw_artist(a)
        for a in self.edge_guides:
            self.ax.draw_artist(a)

        self.blit()
        self.guides_need_draw = False

    def getCurrentLevels(self) -> List[int]:
        shape = self.image.get_array().shape
        x0, x1, y0, y1 = self.extent
        py = (y1 - y0) / shape[0]  # Axes coords
        levels = np.array(
            [guide.get_ydata()[0] / py for guide in self.level_guides[:-1]], dtype=int
        )
        return levels

    def getCurrentTrim(self) -> Tuple[int, int]:
        shape = self.image.get_array().shape
        x0, x1, y0, y1 = self.extent
        px = (x1 - x0) / shape[1]  # Axes coords
        trim = np.array(
            [guide.get_xdata()[0] / px for guide in self.edge_guides], dtype=int
        )

        return np.min(trim), np.max(trim)


class StandardsResultsTable(BasicTable):
    LABELS = ["r²", "Gradient", "Intercept", "Sxy", "LOD (3σ)"]

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(2, 5, parent)
        self.setMinimumSize(QtCore.QSize(0, 1))
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setSortingEnabled(False)
        for r in range(2):
            for c in range(len(StandardsResultsTable.LABELS)):
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(r, c, item)

        for i, label in enumerate(StandardsResultsTable.LABELS):
            self.item(0, i).setText(label)

    def clearResults(self) -> None:
        for i in range(len(StandardsResultsTable.LABELS)):
            self.item(1, i).setText("")

    def updateResults(self, calibration: Calibration) -> None:
        for i, v in enumerate(
            [
                calibration.rsq,
                calibration.gradient,
                calibration.intercept,
                calibration.error,
                (3.0 * calibration.error / calibration.gradient),
            ]
        ):
            item = self.item(1, i)
            item.setText(f"{v:.4f}")


class CalibrationPointsTableModel(NumpyArrayTableModel):
    def __init__(self, calibration: Calibration, parent: QtCore.QObject = None):
        self.calibration = calibration
        if self.calibration.points.size == 0:
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
        if self.calibration.points.size > 0:
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
            self.calibration._points = np.empty((0, 2), dtype=np.float64)
        else:
            self.calibration.points = self.array[:, :2]


class StandardsTable(BasicTableView):
    ROW_LABELS = [c for c in "ABCDEFGHIJKLMNOPQRST"]
    COLUMN_CONC = 0
    COLUMN_COUNT = 1
    COLUMN_WEIGHTS = 2

    def __init__(self, calibration: Calibration, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        # self.setSizePolicy(
        #     QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding
        # )
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        model = CalibrationPointsTableModel(calibration, self)
        self.setModel(model)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
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
