"""This module contains dialogs used in pewpew."""

import copy
from io import BytesIO
from typing import Callable

import numpy as np
from pewlib import Calibration, Config, Laser
from pewlib.config import SpotConfig
from pewlib.process import colocal
from pewlib.process.calc import normalise
from pewlib.process.threshold import otsu
from pewlib.srr import SRRConfig
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction, qToolButton
from pewpew.charts.calibration import CalibrationView
from pewpew.charts.colocal import ColocalisationView
from pewpew.charts.histogram import HistogramView
from pewpew.graphics.imageitems import LaserImageItem
from pewpew.lib import kmeans
from pewpew.lib.pratt import Reducer
from pewpew.models import CalibrationPointsTableModel
from pewpew.validators import (
    ConditionalLimitValidator,
    DecimalValidator,
    DecimalValidatorNoZero,
    DoubleSignificantFiguresDelegate,
    PercentOrDecimalValidator,
)
from pewpew.widgets.ext import CollapsableWidget, ValidColorLineEdit
from pewpew.widgets.modelviews import BasicTableView
from pewpew.widgets.tools.calculator import CalculatorFormula, CalculatorTool
from pewpew.widgets.tools.filtering import FilteringTool


class ApplyDialog(QtWidgets.QDialog):
    """A dialog with Apply, Ok and Close buttons.

    When Apply is pressed the signal `applyPressed` is emitted, with the dialog as an argument.
    Implement `isComplete` and connect `completeChanged` to disable the Apply and Ok buttons
    in specific circumstances.
    Widgets should be added to the `layout_main` layout.
    """

    applyPressed = QtCore.Signal(QtCore.QObject)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.layout_main = QtWidgets.QVBoxLayout()
        self.layout_buttons = QtWidgets.QHBoxLayout()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply,
            self,
        )
        self.button_box.clicked.connect(self.buttonClicked)
        self.layout_buttons.addWidget(self.button_box)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_main)
        layout.addLayout(self.layout_buttons)
        self.setLayout(layout)

    def buttonClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.Apply:
            self.apply()
            self.applyPressed.emit(self)
        elif sb == QtWidgets.QDialogButtonBox.Ok:
            self.apply()
            self.applyPressed.emit(self)
            self.accept()
        else:
            self.reject()

    def apply(self) -> None:
        pass

    def isComplete(self) -> bool:
        return True  # pragma: no cover

    @QtCore.Slot()
    def completeChanged(self) -> None:
        enabled = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(enabled)
        self.button_box.button(QtWidgets.QDialogButtonBox.Apply).setEnabled(enabled)


class CalibrationPointsWidget(CollapsableWidget):
    """Displays calibration points in a table.

    Levels can be added or removed and the weighting set.
    If 'Custom' weighting is used then the weights may be edited.
    """

    levelsChanged = QtCore.Signal(int)
    weightingChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__("Points", parent)

        self.action_add_level = qAction(
            "list-add",
            "Add Level",
            "Adds a level to the calibraiton.",
            self.addCalibrationLevel,
        )
        self.action_remove_level = qAction(
            "list-remove",
            "Remove Level",
            "Removes a level to the calibraiton.",
            self.removeCalibrationLevel,
        )

        self.button_add = qToolButton(action=self.action_add_level)
        self.button_remove = qToolButton(action=self.action_remove_level)

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_weighting.addItems(Calibration.KNOWN_WEIGHTING)
        self.combo_weighting.addItem("Custom")
        self.combo_weighting.currentTextChanged.connect(self.weightingChanged)

        label_weighting = QtWidgets.QLabel("Weighting:")

        self.model = CalibrationPointsTableModel(
            Calibration(),
            orientation=QtCore.Qt.Orientation.Horizontal,
            counts_editable=True,
            parent=self,
        )

        self.levelsChanged.connect(self.updateButtonRemoveEnabled)
        self.model.modelReset.connect(self.updateButtonRemoveEnabled)
        self.combo_weighting.currentTextChanged.connect(self.updateWeighting)

        self.table = BasicTableView()
        self.table.setItemDelegate(DoubleSignificantFiguresDelegate(4))
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.table.setMaximumWidth(800)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )

        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addWidget(self.button_remove, 0, QtCore.Qt.AlignLeft)
        layout_buttons.addWidget(self.button_add, 0, QtCore.Qt.AlignLeft)
        layout_buttons.addStretch(1)

        layout_weighting = QtWidgets.QHBoxLayout()
        layout_weighting.addStretch(1)
        layout_weighting.addWidget(label_weighting, 0, QtCore.Qt.AlignRight)
        layout_weighting.addWidget(self.combo_weighting, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_buttons)
        layout.addWidget(self.table)
        layout.addLayout(layout_weighting)
        self.area.setLayout(layout)

    def setCurrentWeighting(self, weighting: str) -> None:
        self.combo_weighting.blockSignals(True)
        if weighting in Calibration.KNOWN_WEIGHTING:
            self.combo_weighting.setCurrentText(weighting)
        else:
            self.combo_weighting.setCurrentText("Custom")
        self.combo_weighting.blockSignals(False)

    def updateWeighting(self) -> None:
        weighting = self.combo_weighting.currentText()
        self.model.setWeighting(weighting)
        self.weightingChanged.emit(weighting)

    def updateButtonRemoveEnabled(self) -> None:
        columns = self.model.columnCount()
        self.button_remove.setEnabled(columns > 0)

    def addCalibrationLevel(self) -> None:
        columns = self.model.columnCount()
        self.model.insertColumn(columns)
        self.levelsChanged.emit(columns + 1)

    def removeCalibrationLevel(self) -> None:
        columns = self.model.columnCount() - 1
        self.model.removeColumn(columns)
        self.levelsChanged.emit(columns)


class CalibrationDialog(ApplyDialog):
    """A dialog for displaying and editing calibrations."""

    calibrationSelected = QtCore.Signal(dict)
    calibrationApplyAll = QtCore.Signal(dict)

    def __init__(
        self,
        calibrations: dict[str, Calibration],
        current_element: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.calibrations = copy.deepcopy(calibrations)

        self.action_copy = qAction(
            "edit-paste",
            "Copy to Clipboard",
            "Copy the current calibration to the system clipboard.",
            self.copyToClipboard,
        )
        self.action_copy_all = qAction(
            "edit-paste",
            "Copy All to Clipboard",
            "Copy the all calibrations to the system clipboard.",
            self.copyAllToClipboard,
        )
        self.button_copy = qToolButton(action=self.action_copy)
        self.button_copy.addAction(self.action_copy_all)

        # LIne edits
        self.lineedit_gradient = QtWidgets.QLineEdit()
        self.lineedit_gradient.setValidator(DecimalValidatorNoZero(-1e10, 1e10, 4))
        self.lineedit_gradient.setPlaceholderText("1.0000")
        self.lineedit_intercept = QtWidgets.QLineEdit()
        self.lineedit_intercept.setValidator(DecimalValidator(-1e10, 1e10, 4))
        self.lineedit_intercept.setPlaceholderText("0.0000")
        self.lineedit_unit = QtWidgets.QLineEdit()
        self.lineedit_unit.setPlaceholderText("")

        # Element combo
        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.addItems(list(self.calibrations.keys()))
        self.combo_element.setCurrentText(current_element)
        self.previous_index = self.combo_element.currentIndex()
        self.combo_element.currentIndexChanged.connect(self.comboChanged)

        # Check all
        self.check_all = QtWidgets.QCheckBox("Apply calibration to all images.")

        # Button to plot
        self.button_plot = QtWidgets.QPushButton("Plot")
        self.button_plot.setEnabled(self.calibrations[current_element].points.size > 0)
        self.button_plot.pressed.connect(self.showCurve)

        self.points = CalibrationPointsWidget(self)
        self.points.levelsChanged.connect(self.updatePlotEnabled)
        self.points.weightingChanged.connect(self.updateLineEdits)
        self.points.model.dataChanged.connect(self.updateLineEdits)
        self.points.model.dataChanged.connect(self.updatePlotEnabled)

        layout_elements = QtWidgets.QHBoxLayout()
        layout_elements.addWidget(self.button_plot, 0, QtCore.Qt.AlignLeft)
        layout_elements.addWidget(self.combo_element, 0, QtCore.Qt.AlignRight)

        # Form layout for line edits
        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Gradient:", self.lineedit_gradient)
        layout_form.addRow("Intercept:", self.lineedit_intercept)
        layout_form.addRow("Unit:", self.lineedit_unit)

        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_form, 1)
        layout_horz.addWidget(self.button_copy, 0, QtCore.Qt.AlignTop)

        self.layout_main.addLayout(layout_horz)
        self.layout_main.addWidget(self.points)
        self.layout_main.addWidget(self.check_all)
        self.layout_main.addLayout(layout_elements)

        self.updateLineEdits()
        self.updatePoints()

    def apply(self) -> None:
        self.updateCalibration(self.combo_element.currentText())
        if self.check_all.isChecked():
            self.calibrationApplyAll.emit(self.calibrations)
        else:
            self.calibrationSelected.emit(self.calibrations)

    def copyToClipboard(self) -> None:
        """Copy the current calibration to the system clipboard."""
        name = self.combo_element.currentText()
        self.updateCalibration(name)

        text = (
            f"{name}\n"
            f"gradient\t{self.calibrations[name].gradient}\n"
            f"intercept\t{self.calibrations[name].intercept}\n"
            f"unit\t{self.calibrations[name].unit}\n"
        )
        if self.calibrations[name].rsq is not None:
            text += f"rsq\t{self.calibrations[name].rsq}\n"
        if self.calibrations[name].error is not None:
            text += f"error\t{self.calibrations[name].error}\n"
        if self.calibrations[name].points.size > 0:
            x = "\t".join(str(x) for x in self.calibrations[name].x)
            y = "\t".join(str(y) for y in self.calibrations[name].y)
            text += f"points\nx\t{x}\ny\t{y}\n"

        mime = QtCore.QMimeData()
        mime.setText(text)
        with BytesIO() as fp:
            np.savez(fp, name=self.calibrations[name].to_array())
            mime.setData("application/x-pew2calibration", fp.getvalue())
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def copyAllToClipboard(self) -> None:
        """Copy all calibrations to the system clipboard."""
        name = self.combo_element.currentText()
        self.updateCalibration(name)

        names = "\t".join(name for name in self.calibrations)
        gradients = "\t".join(str(c.gradient) for c in self.calibrations.values())
        intercepts = "\t".join(str(c.intercept) for c in self.calibrations.values())
        units = "\t".join(c.unit for c in self.calibrations.values())
        rsqs = "\t".join(str(c.rsq or "") for c in self.calibrations.values())
        errs = "\t".join(str(c.error or "") for c in self.calibrations.values())

        text = (
            f"\t{names}\n"
            f"gradient\t{gradients}\n"
            f"intercept\t{intercepts}\n"
            f"unit\t{units}\n"
            f"rsq\t{rsqs}\n"
            f"error\t{errs}\n"
        )

        mime = QtCore.QMimeData()
        mime.setText(text)
        with BytesIO() as fp:
            np.savez(fp, **{k: v.to_array() for k, v in self.calibrations.items()})
            mime.setData("application/x-pew2calibration", fp.getvalue())
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def comboChanged(self) -> None:
        previous = self.combo_element.itemText(self.previous_index)

        self.updateCalibration(previous)
        self.updateLineEdits()
        self.updatePoints()
        self.updatePlotEnabled()

        self.previous_index = self.combo_element.currentIndex()

    def updatePlotEnabled(self) -> None:
        points = self.calibrations[self.combo_element.currentText()].points
        no_nans = ~np.isnan(points).any(axis=1)
        self.button_plot.setEnabled(np.count_nonzero(no_nans) >= 2)

    def showCurve(self) -> None:
        """Plot the current calibration in a new dialog."""
        dlg = CalibrationCurveDialog(
            self.combo_element.currentText(),
            self.calibrations[self.combo_element.currentText()],
            parent=self,
        )
        dlg.show()

    def updateCalibration(self, name: str) -> None:
        gradient = self.lineedit_gradient.text()
        intercept = self.lineedit_intercept.text()
        unit = self.lineedit_unit.text()

        if gradient != "":
            self.calibrations[name].gradient = float(gradient)
        if intercept != "":
            self.calibrations[name].intercept = float(intercept)
        if unit != "":
            self.calibrations[name].unit = unit

    def updateLineEdits(self) -> None:
        name = self.combo_element.currentText()

        gradient = self.calibrations[name].gradient
        if gradient == 1.0:
            self.lineedit_gradient.clear()
        else:
            self.lineedit_gradient.setText(f"{gradient:.4f}")
        intercept = self.calibrations[name].intercept
        if intercept == 0.0:
            self.lineedit_intercept.clear()
        else:
            self.lineedit_intercept.setText(f"{intercept:.4f}")
        unit = self.calibrations[name].unit
        if unit == "":
            self.lineedit_unit.clear()
        else:
            self.lineedit_unit.setText(str(unit))

    def updatePoints(self) -> None:
        name = self.combo_element.currentText()
        self.points.model.setCalibration(self.calibrations[name], resize=True)
        self.points.setCurrentWeighting(self.calibrations[name].weighting)


class CalibrationCurveDialog(QtWidgets.QDialog):
    """Plots a calibration."""

    def __init__(
        self,
        title: str,
        calibration: Calibration,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration Curve")
        self.chart = CalibrationView(parent=self)
        self.chart.plot.setTitle(title)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chart)
        self.setLayout(layout)

        self.updateChart(calibration)

    def updateChart(self, calibration: Calibration) -> None:
        self.chart.xaxis.setLabel(calibration.unit)
        no_nans = ~np.isnan(calibration.points).any(axis=1)
        points = calibration.points[no_nans]
        self.chart.drawPoints(points)
        self.chart.drawTrendline()


class ColorRangeDialog(ApplyDialog):
    """Dialog for selecting the colortable ranges of an element.

    Args:
        ranges: the current ranges
        default_range: current default range
        elements: availble elements
        current_element: start dialog with this element
        parent: aprent widget
    """

    def __init__(
        self,
        ranges: dict[str, tuple[float | str, float | str]],
        default_range: tuple[float | str, float | str],
        elements: list[str],
        current_element: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.default_range = default_range
        self.ranges = copy.copy(ranges)
        self.previous_element = current_element
        self.setWindowTitle("Colormap Range")

        self.lineedit_min = QtWidgets.QLineEdit()
        self.lineedit_min.setToolTip("Percentile for minium colormap value.")
        self.lineedit_min.setValidator(
            PercentOrDecimalValidator(-1e99, 1e99, parent=self.lineedit_min)
        )
        self.lineedit_max = QtWidgets.QLineEdit()
        self.lineedit_max.setValidator(
            PercentOrDecimalValidator(-1e99, 1e99, parent=self.lineedit_max)
        )
        self.lineedit_max.setToolTip("Percentile for maximum colormap value.")

        # Only add the elements combo if there are any open files
        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.addItems(elements)
        self.combo_element.setCurrentText(self.previous_element)
        self.combo_element.currentIndexChanged.connect(self.comboChanged)
        self.combo_element.setVisible(len(elements) > 0)

        # Checkbox
        self.check_all = QtWidgets.QCheckBox("Apply range to all elements.")
        self.check_all.setChecked(len(elements) == 0)
        self.check_all.setEnabled(len(elements) > 0)
        self.check_all.clicked.connect(self.enableComboElement)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Minimum:", self.lineedit_min)
        layout_form.addRow("Maximum:", self.lineedit_max)

        self.layout_main.addLayout(layout_form)
        self.layout_main.addWidget(self.combo_element, 0, QtCore.Qt.AlignRight)
        self.layout_main.addWidget(self.check_all)

        self.updateLineEdits()

    def enableComboElement(self, enabled: bool) -> None:
        self.combo_element.setEnabled(not enabled)
        self.updateLineEdits()

    def updateLineEdits(self) -> None:
        self.lineedit_min.setPlaceholderText(str(self.default_range[0]))
        self.lineedit_max.setPlaceholderText(str(self.default_range[1]))
        tmin, tmax = "", ""

        # If the combobox is disabled then shown default range as true text
        if self.combo_element.isEnabled():
            # tmin, tmax = "", ""
            # If there is a current element then update text to it's value, if exists
            current_element = self.combo_element.currentText()
            if current_element in self.ranges:
                range = self.ranges[current_element]
                tmin, tmax = str(range[0]), str(range[1])

        else:
            tmin, tmax = str(self.default_range[0]), str(self.default_range[1])

        self.lineedit_min.setText(tmin)
        self.lineedit_max.setText(tmax)

    def comboChanged(self) -> None:
        self.updateRange(self.previous_element)
        self.updateLineEdits()
        self.previous_element = self.combo_element.currentText()

    def updateRange(self, element: str | None = None) -> None:
        tmin, tmax = self.lineedit_min.text(), self.lineedit_max.text()
        vmin, vmax = self.ranges.get(element or "", self.default_range)

        if tmin != "":
            vmin = tmin if "%" in tmin else float(tmin)
        if tmax != "":
            vmax = tmax if "%" in tmax else float(tmax)

        # Unless at least one value is set return
        if tmin == "" and tmax == "":
            return  # pragma: no cover

        if element is not None:
            self.ranges[element] = (vmin, vmax)
        else:
            self.ranges = {}
            self.default_range = (vmin, vmax)

    def apply(self) -> None:
        current_element = self.combo_element.currentText()
        self.updateRange(current_element if self.combo_element.isEnabled() else None)


class ColocalisationDialog(QtWidgets.QDialog):
    """Dialog with colocalisation information for data."""

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray | None = None,
        # colors: list[tuple[float, ...]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        assert data.dtype.names is not None
        super().__init__(parent)
        self.setWindowTitle("Colocalisation")
        self.data = data
        self.mask = mask

        self.chart = ColocalisationView()

        self.combo_name1 = QtWidgets.QComboBox()
        self.combo_name1.addItems(data.dtype.names)
        self.combo_name1.currentIndexChanged.connect(self.refresh)

        self.combo_name2 = QtWidgets.QComboBox()
        self.combo_name2.addItems(data.dtype.names)
        self.combo_name2.setCurrentIndex(1)
        self.combo_name2.currentIndexChanged.connect(self.refresh)

        self.label_r = QtWidgets.QLabel()
        self.label_p = QtWidgets.QLabel()
        self.label_icq = QtWidgets.QLabel()
        self.label_m1 = QtWidgets.QLabel()
        self.label_m2 = QtWidgets.QLabel()

        self.button_p = qToolButton("view-refresh")
        self.button_p.setToolTip("Calculate Pearson r probability.")
        self.button_p.pressed.connect(self.calculatePearsonsProbablity)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.close)

        group_pearson = QtWidgets.QGroupBox("Pearson")
        layout_pearson = QtWidgets.QFormLayout()
        layout_pearson.addRow("r:", self.label_r)
        layout_p = QtWidgets.QHBoxLayout()
        layout_p.addWidget(self.label_p, 1)
        layout_p.addWidget(self.button_p, 0)
        layout_pearson.addRow("ρ:", layout_p)
        group_pearson.setLayout(layout_pearson)

        group_manders = QtWidgets.QGroupBox("Manders")
        layout_manders = QtWidgets.QFormLayout()
        layout_manders.addRow("M1:", self.label_m1)
        layout_manders.addRow("M2:", self.label_m2)
        group_manders.setLayout(layout_manders)

        group_li = QtWidgets.QGroupBox("Li")
        layout_li = QtWidgets.QFormLayout()
        layout_li.addRow("ICQ:", self.label_icq)
        group_li.setLayout(layout_li)

        layout_combos = QtWidgets.QFormLayout()
        layout_combos.addRow("Element 1:", self.combo_name1)
        layout_combos.addRow("Element 2:", self.combo_name2)

        layout_vert = QtWidgets.QVBoxLayout()
        layout_vert.addSpacing(12)
        layout_vert.addWidget(group_pearson)
        layout_vert.addWidget(group_manders)
        layout_vert.addWidget(group_li)
        layout_vert.addStretch(1)
        layout_vert.addLayout(layout_combos)

        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_vert)
        layout_horz.addWidget(self.chart, 1)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_horz)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

        self.refresh()

    def refresh(self) -> None:
        n1 = self.combo_name1.currentText()
        n2 = self.combo_name2.currentText()
        x = self.data[n1].astype(float)
        y = self.data[n2].astype(float)

        if self.mask is not None:
            x, y = x[self.mask], y[self.mask]

        nan_mask = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~nan_mask]
        y = y[~nan_mask]

        x, y = normalise(x), normalise(y)

        # # Pearson
        # r = colocal.pearsonr(x, y)

        # Li
        icq = colocal.li_icq(x, y)

        x, y = x.ravel(), y.ravel()

        # Choose a more approriate threshold?
        # TODO this is really slow, python loops?
        t, a, b = colocal.costes_threshold(x, y)
        tx, ty = t, t * a + b
        r = colocal.pearsonr(
            x[np.logical_and(x > tx, y > ty)], y[np.logical_and(x > tx, y > ty)]
        )
        m1, m2 = colocal.manders(x, y, tx, ty)

        self.label_r.setText(f"{r:.2f}")
        self.label_p.setText("")
        self.label_icq.setText(f"{icq:.2f}")
        self.label_m1.setText(f"{m1:.2f}")
        self.label_m2.setText(f"{m2:.2f}")

        self.button_p.setEnabled(True)

        if x.size > 10000:  # pragma: no cover
            n = np.random.choice(x.size, 10000)
            x, y = x[n], y[n]

        self.chart.clear()
        self.chart.drawPoints(x, y, tx, ty)
        self.chart.drawLine(a, b)
        self.chart.drawThresholds(tx, ty)

        self.chart.xaxis.setLabel(n1)
        self.chart.yaxis.setLabel(n2)

    def calculatePearsonsProbablity(self) -> None:
        x = self.data[self.combo_name1.currentText()]
        y = self.data[self.combo_name2.currentText()]

        _, p = colocal.pearsonr_probablity(x, y, mask=self.mask, n=500)
        self.label_p.setText(f"{p:.2f}")

        self.button_p.setEnabled(False)


class ConfigDialog(ApplyDialog):
    """Dialog view viewing and editing laser configurations."""

    configSelected = QtCore.Signal(Config)
    configApplyAll = QtCore.Signal(Config)

    def __init__(self, config: Config, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.config = copy.copy(config)

        self.action_copy = qAction(
            "edit-paste",
            "Copy to Clipboard",
            "Copy the current configuration to the system clipboard.",
            self.copyToClipboard,
        )
        self.button_copy = qToolButton(action=self.action_copy)

        # Line edits
        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setText(str(self.config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidator(0, 1e9, 4))
        self.lineedit_spotsize.setToolTip("Diameter of the laser spot.")
        self.lineedit_spotsize.textChanged.connect(self.completeChanged)

        if isinstance(self.config, SpotConfig):
            self.lineedit_spotsize_y = QtWidgets.QLineEdit()
            self.lineedit_spotsize_y.setText(str(self.config.spotsize_y))
            self.lineedit_spotsize_y.setValidator(DecimalValidator(0, 1e9, 4))
            self.lineedit_spotsize_y.textChanged.connect(self.completeChanged)

        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setText(str(self.config.speed))
        self.lineedit_speed.setValidator(DecimalValidator(0, 1e9, 4))
        self.lineedit_speed.setToolTip("Scanning speed of the laser.")
        self.lineedit_speed.textChanged.connect(self.completeChanged)

        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setText(str(self.config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidator(0, 1e9, 4))
        self.lineedit_scantime.setToolTip("Total dwell time for one aquistion (pixel).")
        self.lineedit_scantime.textChanged.connect(self.completeChanged)

        if isinstance(self.config, SRRConfig):
            self.lineedit_warmup = QtWidgets.QLineEdit()
            self.lineedit_warmup.setText(str(self.config.warmup))
            self.lineedit_warmup.setValidator(DecimalValidator(0, 1e3, 1))
            self.lineedit_warmup.setToolTip(
                "Laser warmup time; delay before aquisition."
            )
            self.lineedit_warmup.textChanged.connect(self.completeChanged)
            self.spinbox_offsets = QtWidgets.QSpinBox()
            self.spinbox_offsets.setRange(2, 10)
            self.spinbox_offsets.setValue(self.config._subpixel_size)
            self.spinbox_offsets.setToolTip(
                "Number of subpixels per pixel. "
                "Offseting each layer by 50% will have a subpixel width of 2."
            )

        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")

        # Form layout for line edits
        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Spotsize (μm):", self.lineedit_spotsize)
        if isinstance(config, SpotConfig):
            layout_form.addRow("Spotsize Y (μm):", self.lineedit_spotsize_y)
        else:
            layout_form.addRow("Speed (μm/s):", self.lineedit_speed)
            layout_form.addRow("Scantime (s):", self.lineedit_scantime)
        if isinstance(config, SRRConfig):
            layout_form.addRow("Warmup (s):", self.lineedit_warmup)
            layout_form.addRow("Subpixel width:", self.spinbox_offsets)

        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_form, 1)
        layout_horz.addWidget(self.button_copy, 0, QtCore.Qt.AlignTop)

        self.layout_main.addLayout(layout_horz)
        self.layout_main.addWidget(self.check_all)

    def updateConfig(self) -> None:
        self.config.spotsize = float(self.lineedit_spotsize.text())
        self.config.speed = float(self.lineedit_speed.text())
        self.config.scantime = float(self.lineedit_scantime.text())
        if isinstance(self.config, SpotConfig):
            self.config.spotsize_y = float(self.lineedit_spotsize_y.text())
        if isinstance(self.config, SRRConfig):
            self.config.warmup = float(self.lineedit_warmup.text())
            self.config.set_equal_subpixel_offsets(self.spinbox_offsets.value())

    def apply(self) -> None:
        self.updateConfig()
        if self.check_all.isChecked():
            self.configApplyAll.emit(self.config)
        else:
            self.configSelected.emit(self.config)

    def copyToClipboard(self) -> None:
        self.updateConfig()
        text = (
            f"spotsize\t{self.config.spotsize}\n"
            f"speed\t{self.config.speed}\n"
            f"scantime\t{self.config.scantime}\n"
        )
        if isinstance(self.config, SRRConfig):
            text += f"warmup\t{self.config.warmup}\nsubpixel offsets\t"
            text += "\t".join(str(o) for o in self.config.subpixel_offsets)

        mime = QtCore.QMimeData()
        mime.setText(text)
        with BytesIO() as fp:
            np.save(fp, self.config.to_array())
            mime.setData("application/x-pew2config", fp.getvalue())
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def isComplete(self) -> bool:
        if not self.lineedit_spotsize.hasAcceptableInput():
            return False
        if not self.lineedit_speed.hasAcceptableInput():
            return False
        if not self.lineedit_scantime.hasAcceptableInput():
            return False
        if isinstance(self.config, SRRConfig):
            if not self.lineedit_warmup.hasAcceptableInput():
                return False
        return True


class InformationDialog(QtWidgets.QDialog):
    """Dialog for viewing and editing laser informations."""

    infoChanged = QtCore.Signal(dict)

    read_only_items = ["Name", "File Path", "File Version", "Processing"]

    def __init__(self, info: dict[str, str], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.setMinimumSize(400, 400)
        self.setWindowTitle("Information")

        self.layout_info = QtWidgets.QVBoxLayout()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.table = QtWidgets.QTableWidget(len(info), 2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )

        self.setInformation(info)
        self.ensureEmptyRow()

        self.table.resizeColumnsToContents()
        self.table.itemChanged.connect(self.ensureEmptyRow)
        self.table.itemChanged.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def accept(self) -> None:
        info = self.information()
        self.infoChanged.emit(info)
        super().accept()

    @QtCore.Slot()
    def completeChanged(self) -> None:
        enabled = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(enabled)

    @QtCore.Slot()
    def ensureEmptyRow(self) -> None:
        row = self.table.rowCount()
        self.table.blockSignals(True)
        if (
            self.table.item(row - 1, 0).text() != ""
            or self.table.item(row - 1, 1).text() != ""
        ):
            self.table.setRowCount(row + 1)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
        self.table.blockSignals(False)

    def isComplete(self) -> bool:
        keys = [self.table.item(i, 0).text() for i in range(self.table.rowCount())]
        keys = list(filter("".__ne__, keys))  # Remove empty
        return len(keys) == len(set(keys))  # Ensure all unqiue

    def setInformation(self, info: dict[str, str]) -> None:
        for i, (key, val) in enumerate(info.items()):
            items = QtWidgets.QTableWidgetItem(key), QtWidgets.QTableWidgetItem(val)
            if key in InformationDialog.read_only_items:
                [i.setFlags(i.flags() & ~QtCore.Qt.ItemIsEnabled) for i in items]
            self.table.setItem(i, 0, items[0])
            self.table.setItem(i, 1, items[1])

    def information(self) -> dict[str, str]:
        info = {}
        for i in range(self.table.rowCount()):
            key = self.table.item(i, 0).text()
            if key != "":
                info[key] = self.table.item(i, 1).text()
        return info


class NameEditDialog(QtWidgets.QDialog):
    """Dialog for editing the current laser element names."""

    originalNameRole = QtCore.Qt.UserRole + 1
    namesSelected = QtCore.Signal(dict)

    def __init__(
        self,
        names: list[str],
        allow_remove: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit Names")

        self.allow_remove = allow_remove

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.list = QtWidgets.QListWidget()
        self.list.itemDoubleClicked.connect(self.list.editItem)
        self.addNames(names)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def accept(self) -> None:
        items = [self.list.item(i) for i in range(self.list.count())]
        rename = {}
        if self.allow_remove and all(
            item.checkState() == QtCore.Qt.CheckState.Unchecked for item in items
        ):
            QtWidgets.QMessageBox.warning(
                self, "Name Change Dialog", "Unable to remove all elements."
            )
            return
        for item in items:
            if not self.allow_remove or (
                item.flags() & QtCore.Qt.ItemIsUserCheckable
                and item.checkState() == QtCore.Qt.Checked
            ):
                rename[item.data(NameEditDialog.originalNameRole)] = item.text()
        self.namesSelected.emit(rename)
        super().accept()

    def addName(self, name: str) -> None:
        item = QtWidgets.QListWidgetItem(self.list)
        item.setText(name)
        item.setData(NameEditDialog.originalNameRole, name)
        item.setFlags(QtCore.Qt.ItemIsEditable | item.flags())
        if self.allow_remove:
            item.setCheckState(QtCore.Qt.Checked)
        self.list.addItem(item)

    def addNames(self, names: list[str]) -> None:
        for name in names:
            self.addName(name)


class PixelSizeDialog(ApplyDialog):
    sizeSelected = QtCore.Signal(QtCore.QSizeF)

    def __init__(self, size: QtCore.QSizeF, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Set Pixel Size")

        self.xsize = QtWidgets.QLineEdit(str(size.width()))
        self.xsize.setValidator(DecimalValidator(0.001, 999.999, 3))

        self.ysize = QtWidgets.QLineEdit(str(size.height()))
        self.ysize.setValidator(DecimalValidator(0.001, 999.999, 3))

        layoutx = QtWidgets.QHBoxLayout()
        layoutx.addWidget(self.xsize, 1)
        layoutx.addWidget(QtWidgets.QLabel("μm"), 0, QtCore.Qt.AlignRight)
        layouty = QtWidgets.QHBoxLayout()
        layouty.addWidget(self.ysize, 1)
        layouty.addWidget(QtWidgets.QLabel("μm"), 0, QtCore.Qt.AlignRight)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("x:", layoutx)
        layout_form.addRow("y:", layouty)

        self.layout_main.addLayout(layout_form)

    def size(self) -> QtCore.QSizeF:
        return QtCore.QSizeF(float(self.xsize.text()), float(self.ysize.text()))

    def apply(self) -> None:
        self.sizeSelected.emit(self.size())


class ProcessItemWidget(QtWidgets.QWidget):
    completeChanged = QtCore.Signal()
    closeRequested = QtCore.Signal(QtWidgets.QWidget)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.action_close = qAction(
            "view-close",
            "Close Process",
            "Remove this process from the pipeline.",
            lambda: self.closeRequested.emit(self),
        )
        self.button_close = qToolButton(action=self.action_close)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        return False


class ProcessCalculatorItemWidget(ProcessItemWidget):
    def __init__(
        self,
        names: list[str],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.lineedit_name = QtWidgets.QLineEdit()
        self.lineedit_name.textChanged.connect(self.completeChanged)
        self.lineedit_expr = QtWidgets.QLineEdit()
        self.lineedit_expr.textChanged.connect(self.completeChanged)

        layout_controls = QtWidgets.QHBoxLayout()
        layout_controls.addWidget(QtWidgets.QLabel("Name:"))
        layout_controls.addWidget(self.lineedit_name)
        layout_controls.addWidget(QtWidgets.QLabel("Expr:"))
        layout_controls.addWidget(self.lineedit_expr)
        self.layout().insertLayout(0, layout_controls, 1)

    @property
    def name(self) -> str:
        return self.lineedit_name.text()

    @property
    def expr(self) -> str:
        return self.lineedit_expr.text()

    def isComplete(self) -> bool:
        for le in [self.lineedit_name, self.lineedit_expr]:
            if le.text() == "" or not le.hasAcceptableInput():
                return False
        return True


class ProcessFilterItemWidget(ProcessItemWidget):
    def __init__(
        self,
        names: list[str],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.combo_filter = QtWidgets.QComboBox()
        self.combo_filter.addItems(FilteringTool.methods.keys())
        self.combo_filter.setCurrentText("Local Median")
        self.combo_filter.activated.connect(self.filterChanged)

        self.combo_names = QtWidgets.QComboBox()
        self.combo_names.addItems(names)
        self.combo_names.addItem("*")

        nparams = np.amax([len(f["params"]) for f in FilteringTool.methods.values()])
        self.label_fparams = [QtWidgets.QLabel() for _ in range(nparams)]
        self.lineedit_fparams = [ValidColorLineEdit() for _ in range(nparams)]
        for le in self.lineedit_fparams:
            le.setValidator(ConditionalLimitValidator(0.0, 0.0, 4, condition=None))
            le.textChanged.connect(self.completeChanged)

        layout_controls = QtWidgets.QHBoxLayout()
        layout_controls.addWidget(self.combo_filter)
        layout_controls.addWidget(self.combo_names)
        for i in range(len(self.label_fparams)):
            layout_controls.addWidget(
                self.label_fparams[i], 0, QtCore.Qt.AlignmentFlag.AlignRight
            )
            layout_controls.addWidget(self.lineedit_fparams[i])

        self.layout().insertLayout(0, layout_controls, 1)
        self.filterChanged()

    @property
    def method(self) -> str:
        return self.combo_filter.currentText()

    @property
    def name(self) -> str | None:
        name = self.combo_names.currentText()
        if name == "*":
            return None
        return name

    @property
    def fparams(self) -> list[float]:
        return [float(le.text()) for le in self.lineedit_fparams if le.isEnabled()]

    def isComplete(self) -> bool:
        for le in self.lineedit_fparams:
            if le.isEnabled() and not le.hasAcceptableInput():
                return False
        return True

    def filterChanged(self) -> None:
        filter_ = FilteringTool.methods[self.combo_filter.currentText()]
        # Clear all the current params
        for le in self.label_fparams:
            le.setVisible(False)
        for le in self.lineedit_fparams:
            le.setVisible(False)

        params: list[tuple[str, float, tuple, Callable[[float], bool]]] = filter_[
            "params"
        ]

        for i, (symbol, default, range, condition) in enumerate(params):
            self.label_fparams[i].setText(f"{symbol}:")
            self.label_fparams[i].setVisible(True)
            self.lineedit_fparams[i].validator().setRange(range[0], range[1], 4)
            self.lineedit_fparams[i].validator().setCondition(condition)
            self.lineedit_fparams[i].setVisible(True)
            self.lineedit_fparams[i].setToolTip(filter_["desc"][i])
            # keep input that's still valid
            if not self.lineedit_fparams[i].hasAcceptableInput():
                self.lineedit_fparams[i].setText(str(default))
                self.lineedit_fparams[i].revalidate()


class ProcessingDialog(QtWidgets.QDialog):
    def __init__(
        self,
        names: list[str],
        items: list[LaserImageItem] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Processing Pipeline")
        self.setMinimumWidth(600)

        self.names = names
        self.items = items or []

        self.action_add_calculator = qAction(
            "list-add",
            "Calculator Process",
            "Add a new process to the pipeline.",
            self.addCalculatorProcess,
        )
        self.action_add_filter = qAction(
            "list-add",
            "Filter Process",
            "Add a new process to the pipeline.",
            self.addFilterProcess,
        )
        self.list = QtWidgets.QListWidget()
        # do manually as has to be emitted after item widget is set
        # self.list.model().rowsInserted.connect(self.completeChanged)
        self.list.model().rowsRemoved.connect(self.completeChanged)

        self.action_add_laser = qAction(
            "list-add",
            "Add Image",
            "Add image for processing via pipeline.",
            self.addApplyLaser,
        )
        self.action_add_all_laser = qAction(
            "list-add",
            "Add All Images",
            "Add all open images to the processing pipeline.",
            self.addAllApplyLaser,
        )
        self.apply_list = QtWidgets.QListWidget()
        self.apply_list.model().rowsInserted.connect(self.completeChanged)
        self.apply_list.model().rowsRemoved.connect(self.completeChanged)

        self.button_load_from_laser = QtWidgets.QPushButton(
            QtGui.QIcon.fromTheme("document-open"), "Load From Laser"
        )
        self.button_load_from_laser.pressed.connect(self.loadProcessingFromLaser)
        self.button_load_from_laser.setEnabled(len(self.items) > 0)

        self.button_add = qToolButton(action=self.action_add_calculator)
        self.button_add.addAction(self.action_add_filter)

        self.button_add_laser = qToolButton(action=self.action_add_laser)
        self.button_add_laser.addAction(self.action_add_all_laser)
        self.button_add_laser.setEnabled(len(self.items) > 0)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok,
            self,
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(
            self.button_load_from_laser, 0, QtCore.Qt.AlignmentFlag.AlignLeft
        )
        button_layout.addWidget(self.button_box, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        process_layout = QtWidgets.QHBoxLayout()
        process_layout.addWidget(QtWidgets.QLabel("Add process:"), 1)
        process_layout.addWidget(self.button_add, 4, QtCore.Qt.AlignmentFlag.AlignLeft)

        apply_layout = QtWidgets.QHBoxLayout()
        apply_layout.addWidget(QtWidgets.QLabel("Apply to:"), 1)
        apply_layout.addWidget(
            self.button_add_laser, 4, QtCore.Qt.AlignmentFlag.AlignLeft
        )

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(process_layout)
        layout.addWidget(self.list)
        layout.addLayout(apply_layout)
        layout.addWidget(self.apply_list)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def addApplyLaser(self) -> None:
        laser_item = self.dialogLoadFromLaser()
        if laser_item is None:
            return
        item = QtWidgets.QListWidgetItem(laser_item.name())
        self.apply_list.addItem(item)

    def addAllApplyLaser(self) -> None:
        for laser_item in self.items:
            item = QtWidgets.QListWidgetItem(laser_item.name())
            self.apply_list.addItem(item)

    def addCalculatorProcess(self) -> ProcessCalculatorItemWidget:
        widget = ProcessCalculatorItemWidget(self.names)
        widget.completeChanged.connect(self.completeChanged)
        widget.closeRequested.connect(self.removeProcess)
        item = QtWidgets.QListWidgetItem()
        self.list.insertItem(self.list.count(), item)
        self.list.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())
        self.completeChanged()
        return widget

    def addFilterProcess(self) -> ProcessFilterItemWidget:
        widget = ProcessFilterItemWidget(self.names)
        widget.completeChanged.connect(self.completeChanged)
        widget.closeRequested.connect(self.removeProcess)
        item = QtWidgets.QListWidgetItem()
        self.list.insertItem(self.list.count(), item)
        self.list.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())
        self.completeChanged()
        return widget

    def removeProcess(self, widget: ProcessFilterItemWidget) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if self.list.itemWidget(item) == widget:
                self.list.takeItem(i)
                break

    def applyPipelineToLaser(self, laser: Laser) -> bool:
        update_required = False
        data = laser.get(flat=True, calibrated=False)
        reducer = Reducer(variables={name: data[name] for name in data.dtype.names})
        reducer.operations.update(
            {k: v[1] for k, v in CalculatorTool.functions.items()}
        )
        for i in range(self.list.count()):
            proc = self.list.itemWidget(self.list.item(i))
            if isinstance(proc, ProcessFilterItemWidget):
                FilteringTool.filterLaser(laser, proc.name, proc.method, proc.fparams)
            elif isinstance(proc, ProcessCalculatorItemWidget):
                calc = reducer.reduce(proc.expr)
                if proc.name in laser.elements:
                    laser.data[proc.name] = calc
                else:
                    update_required = True
                    laser.add(proc.name, calc)
            else:
                raise ValueError("unknown process item type")
        return update_required

    def loadFromString(self, proc_string: str) -> None:
        processes = proc_string.split(";")
        for proc in processes:
            if len(proc) == 0:
                continue
            proc_type = proc[: proc.find("(")]
            proc_params = proc[proc.find("(") + 1 : proc.find(")")]
            if proc_type == "Calculator":
                widget = self.addCalculatorProcess()
                oname, expr = proc_params.split(",")
                widget.lineedit_name.setText(oname)
                widget.lineedit_expr.setText(expr)
            elif proc_type == "Filter":
                widget = self.addFilterProcess()
                name, filter_type, *filter_pstr = proc_params.split(",")
                widget.combo_names.setCurrentText(name)
                widget.combo_filter.setCurrentText(filter_type)
                widget.filterChanged()
                for pstr, le in zip(
                    filter_pstr,
                    [le for le in widget.lineedit_fparams if le.isVisible()],
                ):
                    _, p = pstr.split("=")
                    le.setText(p)
            else:
                raise ValueError(f"unknown processing type '{proc_type}'")

    def dialogLoadFromLaser(self) -> LaserImageItem | None:
        item_names = {item.name(): item for item in self.items}
        name, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Laser", "Laser item:", list(item_names.keys()), editable=False
        )
        if ok and name is not None:
            return item_names[name]
        return None

    def completeChanged(self) -> None:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(self.isComplete())

    def isComplete(self) -> bool:
        if self.list.count() == 0 or self.apply_list.count() == 0:
            return False
        for i in range(self.list.count()):
            widget = self.list.itemWidget(self.list.item(i))
            if widget is not None and not widget.isComplete():
                return False
        return True

    def loadProcessingFromLaser(self) -> None:
        item = self.dialogLoadFromLaser()
        if item is not None and "Processing" in item.laser.info:
            self.loadFromString(item.laser.info["Processing"])

    def accept(self) -> None:
        item_names = {item.name(): item for item in self.items}
        for i in range(self.apply_list.count()):
            item = item_names[self.apply_list.item(i).text()]
            update_required = self.applyPipelineToLaser(item.laser)
            if update_required:
                item.elementsChanged.emit()
            else:
                item.redraw()

        super().accept()


class SelectionDialog(ApplyDialog):
    """Dialog for theshold based selection of data."""

    maskSelected = QtCore.Signal(np.ndarray, "QStringList")

    METHODS = {
        "Manual": (None, None),
        "Mean": (np.mean, None),
        "Median": (np.median, None),
        "Otsu": (otsu, None),
        "K-means": (kmeans.thresholds, ("k: ", 3, (2, 9))),
    }
    COMPARISION = {">": np.greater, "<": np.less, "=": np.equal}

    def __init__(self, item: LaserImageItem, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Selection")

        self.item = item

        self.threshold: float = 0.0

        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(list(self.METHODS.keys()))
        self.combo_method.activated.connect(self.refresh)

        self.spinbox_method = QtWidgets.QSpinBox()
        self.spinbox_method.setEnabled(False)
        self.spinbox_method.valueChanged.connect(self.refresh)

        self.combo_comparison = QtWidgets.QComboBox()
        self.combo_comparison.addItems(list(self.COMPARISION.keys()))
        self.combo_comparison.activated.connect(self.refresh)

        self.spinbox_comparison = QtWidgets.QSpinBox()
        self.spinbox_comparison.setEnabled(False)
        self.spinbox_comparison.setPrefix("t: ")
        self.spinbox_comparison.valueChanged.connect(self.refresh)

        self.lineedit_manual = QtWidgets.QLineEdit("0.0")
        self.lineedit_manual.setValidator(DecimalValidator(-1e99, 1e99, 4))
        self.lineedit_manual.setEnabled(True)
        self.lineedit_manual.textEdited.connect(self.refresh)

        self.check_limit_selection = QtWidgets.QCheckBox(
            "Intersect selection with current selection."
        )
        self.check_limit_threshold = QtWidgets.QCheckBox(
            "Limit thresholding to selected values."
        )
        self.check_limit_threshold.clicked.connect(self.refresh)

        layout_method = QtWidgets.QHBoxLayout()
        layout_method.addWidget(self.combo_method)
        layout_method.addWidget(self.spinbox_method)

        layout_comparison = QtWidgets.QHBoxLayout()
        layout_comparison.addWidget(self.combo_comparison)
        layout_comparison.addWidget(self.spinbox_comparison)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Method", layout_method)
        layout_form.addRow("Comparison", layout_comparison)
        layout_form.addRow("Value", self.lineedit_manual)
        layout_form.addRow(self.check_limit_selection)
        layout_form.addRow(self.check_limit_threshold)

        self.layout_main.addLayout(layout_form)

    def refresh(self) -> None:
        method = self.combo_method.currentText()
        data = self.item.raw_data
        if data is None or len(data) == 0:
            return
        if self.check_limit_threshold.isChecked() and self.item.mask_image is not None:
            data = data[self.item.mask]

        # Remove nans
        data = data[~np.isnan(data)]

        op, var = SelectionDialog.METHODS[method]

        # Compute new threshold
        if method == "Manual":
            self.lineedit_manual.setEnabled(True)
            self.spinbox_method.setEnabled(False)
            self.spinbox_comparison.setEnabled(False)
            self.threshold = float(self.lineedit_manual.text() or np.inf)
        else:
            self.lineedit_manual.setEnabled(False)
            if var is not None:
                self.spinbox_method.setRange(*var[2])
                if not self.spinbox_method.isEnabled():  # First show
                    self.spinbox_method.setValue(var[1])
                    self.spinbox_method.setEnabled(True)

                self.spinbox_method.setPrefix(var[0])
                self.spinbox_comparison.setEnabled(True)
                self.spinbox_comparison.setRange(1, self.spinbox_method.value() - 1)

                self.threshold = op(data, self.spinbox_method.value())[
                    self.spinbox_comparison.value() - 1
                ]
            else:
                self.spinbox_method.setEnabled(False)
                self.spinbox_comparison.setEnabled(False)

                self.threshold = op(data)
            self.lineedit_manual.setText(f"{self.threshold:.4g}")

    def apply(self) -> None:
        comparison = self.COMPARISION[self.combo_comparison.currentText()]
        data = self.item.raw_data
        if data is None:
            return
        mask = comparison(data, self.threshold)
        state = "intersect" if self.check_limit_selection.isChecked() else None
        self.maskSelected.emit(mask, [state])
        self.refresh()


class StatsDialog(QtWidgets.QDialog):
    """Dialog for viewing data statistics.

    Args:
        data: structured array of elements
        mask: mask for input, shape shape as `x`
        units: dict mapping data names to a str
        element: display this element at open
        pixel_size: size of a pixel in μm, for area
        parent: parent widget
    """

    element_changed = QtCore.Signal(str)

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        units: dict[str, str],
        element: str,
        pixel_size: tuple[float, float] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Statistics")

        self.data = self.prepareData(data, mask)
        self.units = units
        self.pixel_size = pixel_size

        self.chart = HistogramView()

        self.button_clipboard = QtWidgets.QPushButton("Copy to Clipboard")
        self.button_clipboard.pressed.connect(self.copyToClipboard)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.close)
        self.button_box.addButton(
            self.button_clipboard, QtWidgets.QDialogButtonBox.ActionRole
        )

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.addItems(self.data.dtype.names or [element])
        self.combo_element.setCurrentText(element)
        self.combo_element.currentIndexChanged.connect(self.updateStats)
        self.combo_element.currentTextChanged.connect(self.element_changed)

        self.label_shape = QtWidgets.QLabel()
        self.label_size = QtWidgets.QLabel()
        self.label_area = QtWidgets.QLabel()

        self.label_min = QtWidgets.QLabel()
        self.label_max = QtWidgets.QLabel()
        self.label_mean = QtWidgets.QLabel()
        self.label_median = QtWidgets.QLabel()
        self.label_stddev = QtWidgets.QLabel()

        stats_left = QtWidgets.QFormLayout()
        stats_left.addRow("Shape:", self.label_shape)
        stats_left.addRow("Size:", self.label_size)
        stats_left.addRow("Area:", self.label_area)

        stats_right = QtWidgets.QFormLayout()
        stats_right.addRow("Min:", self.label_min)
        stats_right.addRow("Max:", self.label_max)
        stats_right.addRow("Mean:", self.label_mean)
        stats_right.addRow("Median:", self.label_median)
        stats_right.addRow("Std Dev:", self.label_stddev)

        stats_box = QtWidgets.QGroupBox()
        stats_layout = QtWidgets.QHBoxLayout()
        stats_layout.addLayout(stats_left)
        stats_layout.addLayout(stats_right)
        stats_box.setLayout(stats_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chart)
        layout.addWidget(stats_box)
        layout.addWidget(self.combo_element, 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        # Calculate the range
        self.updateStats()

    def copyToClipboard(self) -> None:
        data = (
            '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
            "<table>"
        )
        text = ""
        size = self.data[~np.isnan(self.data[self.data.dtype.names[0]])].size
        area = size * self.pixel_size[0] * self.pixel_size[1]

        data += f"<tr><td>Size</td><td>{size}</td></tr>"
        text += f"Size\t{size}\n"
        data += f"<tr><td>Area</td><td>{area}</td><td>μm²</td></tr>"
        text += f"Shape\t{area}\n"

        data += (
            "<tr><td>Name</td><td>Unit</td><td>Min</td><td>Max</td><td>Mean</td>"
            "<td>Median</td><td>Std</tr>"
        )
        text += "Name\tUnit\tMin\tMax\tMean\tMedian\tStd\n"

        for name in self.data.dtype.names:
            nd = self.data[name]
            unit = self.units.get(str(name), "")
            nd = nd[~np.isnan(nd)]

            data += (
                f"<tr><td>{name}</td><td>{unit}</td><td>{np.min(nd)}</td>"
                f"<td>{np.max(nd)}</td><td>{np.mean(nd)}</td><td>{np.median(nd)}</td>"
                f"<td>{np.std(nd)}</td></tr>"
            )
            text += f"{name}\t{unit}\t{np.min(nd)}\t{np.max(nd)}\t"
            f"{np.mean(nd)}\t{np.median(nd)}\t{np.std(nd)}\n"

        text = text.rstrip("\n")
        data += "</table>"

        mime = QtCore.QMimeData()
        mime.setHtml(data)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def updateStats(self) -> None:
        element = self.combo_element.currentText()
        data = self.data[element]
        unit = self.units.get(element, "")

        self.label_shape.setText(str(data.shape))
        self.label_size.setText(str(data.size))
        # Discard nans and shape
        data = data[~np.isnan(data)].ravel()
        if self.pixel_size is not None:
            area = data.size * self.pixel_size[0] * self.pixel_size[1]
            if area > 1e11:
                area /= 1e8
                areaunit = "cm"
            elif area > 1e6:
                area /= 1e6
                areaunit = "mm"
            else:
                areaunit = "μm"

            self.label_area.setText(f"{area:.6g} {areaunit}²")

        self.label_min.setText(f"{np.min(data):.4g} {unit}")
        self.label_max.setText(f"{np.max(data):.4g} {unit}")
        self.label_mean.setText(f"{np.mean(data):.4g} {unit}")
        self.label_median.setText(f"{np.median(data):.4g} {unit}")
        self.label_stddev.setText(f"{np.std(data):.4g} {unit}")

        self.chart.clear()
        self.chart.setHistogram(data.astype(float))
        self.chart.setDataLimits(xMin=-0.05, xMax=1.05, yMax=1.05)

    def isCalibrate(self) -> bool:
        return False  # pragma: no cover

    def prepareData(self, structured: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ix, iy = np.nonzero(mask)
        x0, x1, y0, y1 = np.min(ix), np.max(ix) + 1, np.min(iy), np.max(iy) + 1

        data = np.empty((x1 - x0, y1 - y0), dtype=structured.dtype)
        for name in structured.dtype.names:
            data[name] = np.where(
                mask[x0:x1, y0:y1],
                structured[name][x0:x1, y0:y1],
                np.nan,
            )
        return data
