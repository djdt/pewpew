import copy
import io
import numpy as np

from PySide2 import QtCore, QtWidgets

from pewlib import Calibration, Config
from pewlib.process import colocal
from pewlib.process.calc import normalise
from pewlib.process.threshold import otsu
from pewlib.srr import SRRConfig

from pewpew.actions import qAction, qToolButton
from pewpew.lib import kmeans
from pewpew.validators import (
    DecimalValidator,
    DecimalValidatorNoZero,
    PercentOrDecimalValidator,
)

from pewpew.charts.calibration import CalibrationChart
from pewpew.charts.colocal import ColocalisationChart
from pewpew.charts.histogram import HistogramChart

from pewpew.graphics.lasergraphicsview import LaserGraphicsView

from typing import Dict, List, Tuple, Union


class ApplyDialog(QtWidgets.QDialog):

    applyPressed = QtCore.Signal(QtCore.QObject)

    def __init__(self, parent: QtWidgets.QWidget = None):
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


class CalibrationDialog(ApplyDialog):
    calibrationSelected = QtCore.Signal(dict)
    calibrationApplyAll = QtCore.Signal(dict)

    def __init__(
        self,
        calibrations: Dict[str, Calibration],
        current_isotope: str,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.calibrations = copy.deepcopy(calibrations)

        self.action_copy = qAction(
            "edit-paste",
            "Copy to Clipboard",
            "Copy the current configuration to the system clipboard.",
            self.copy,
        )
        self.button_copy = qToolButton(action=self.action_copy)

        # LIne edits
        self.lineedit_gradient = QtWidgets.QLineEdit()
        self.lineedit_gradient.setValidator(DecimalValidatorNoZero(-1e10, 1e10, 4))
        self.lineedit_gradient.setPlaceholderText("1.0000")
        self.lineedit_intercept = QtWidgets.QLineEdit()
        self.lineedit_intercept.setValidator(DecimalValidator(-1e10, 1e10, 4))
        self.lineedit_intercept.setPlaceholderText("0.0000")
        self.lineedit_unit = QtWidgets.QLineEdit()
        self.lineedit_unit.setPlaceholderText("")

        # Isotope combo
        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.addItems(list(self.calibrations.keys()))
        self.combo_isotope.setCurrentText(current_isotope)
        self.previous_index = self.combo_isotope.currentIndex()
        self.combo_isotope.currentIndexChanged.connect(self.comboChanged)

        # Check all
        self.check_all = QtWidgets.QCheckBox("Apply calibration to all images.")

        # Button to plot
        self.button_plot = QtWidgets.QPushButton("Plot")
        self.button_plot.setEnabled(self.calibrations[current_isotope].points.size > 0)
        self.button_plot.pressed.connect(self.showCurve)

        layout_isotopes = QtWidgets.QHBoxLayout()
        layout_isotopes.addWidget(self.button_plot, 0, QtCore.Qt.AlignLeft)
        layout_isotopes.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignRight)

        # Form layout for line edits
        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Gradient:", self.lineedit_gradient)
        layout_form.addRow("Intercept:", self.lineedit_intercept)
        layout_form.addRow("Unit:", self.lineedit_unit)

        layout_options = QtWidgets.QHBoxLayout()
        layout_options.addWidget(self.check_all, 1)
        layout_options.addWidget(self.button_copy, 0, QtCore.Qt.AlignRight)

        self.layout_main.addLayout(layout_form)
        self.layout_main.addLayout(layout_isotopes)
        self.layout_main.addLayout(layout_options)

        self.updateLineEdits()

    def copy(self) -> None:
        name = self.combo_isotope.currentText()
        self.updateCalibration(name)

        text = (
            f"gradient\t{self.calibrations[name].gradient}\n"
            f"intercept\t{self.calibrations[name].intercept}\n"
            f"unit\t{self.calibrations[name].unit}\n"
        )
        if self.calibrations[name].points.size > 0:
            x = '\t'.join(self.calibrations[name].x)
            y = '\t'.join(self.calibrations[name].y)
            text += f"points\nx\t{x}\ny\t{y}\n"

        mime = QtCore.QMimeData()
        mime.setText()
        with io.BytesIO() as fp:
            np.save(fp, {k: v.to_array() for k, v in self.calibrations.items()})
            mime.setData("application/x-pew2calibration", fp.getvalue())
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def updateLineEdits(self) -> None:
        name = self.combo_isotope.currentText()

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

    def comboChanged(self) -> None:
        previous = self.combo_isotope.itemText(self.previous_index)
        self.updateCalibration(previous)
        self.updateLineEdits()
        self.previous_index = self.combo_isotope.currentIndex()
        self.button_plot.setEnabled(
            self.calibrations[self.combo_isotope.currentText()].points.size > 0
        )

    def showCurve(self) -> None:
        dlg = CalibrationCurveDialog(
            self.combo_isotope.currentText(),
            self.calibrations[self.combo_isotope.currentText()],
            parent=self,
        )
        dlg.show()

    def apply(self) -> None:
        self.updateCalibration(self.combo_isotope.currentText())
        if self.check_all.isChecked():
            self.calibrationApplyAll.emit(self.calibrations)
        else:
            self.calibrationSelected.emit(self.calibrations)


class CalibrationCurveDialog(QtWidgets.QDialog):
    def __init__(
        self, title: str, calibration: Calibration, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration Curve")
        self.chart = CalibrationChart(title, parent=self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chart)
        self.setLayout(layout)

        self.updateChart(calibration)

    def updateChart(self, calibration: Calibration) -> None:
        self.chart.xaxis.setTitleText(calibration.unit)
        self.chart.setPoints(calibration.points)
        self.chart.setLine(
            0.0,
            np.nanmax(calibration.x) * 1.1,
            calibration.gradient,
            calibration.intercept,
        )
        text = f"{calibration.gradient:.4f} × x + {calibration.intercept:.4f}"
        if calibration.rsq is not None:
            text += f"\nr² = {calibration.rsq:.4f}"
        self.chart.setText(text)


class ColorRangeDialog(ApplyDialog):
    def __init__(
        self,
        ranges: Dict[str, Tuple[Union[float, str], Union[float, str]]],
        default_range: Tuple[Union[float, str], Union[float, str]],
        isotopes: List[str],
        current_isotope: str = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.default_range = default_range
        self.ranges = copy.copy(ranges)
        self.previous_isotope = current_isotope
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

        # Only add the isotopes combo if there are any open files
        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.addItems(isotopes)
        self.combo_isotope.setCurrentText(self.previous_isotope)
        self.combo_isotope.currentIndexChanged.connect(self.comboChanged)
        self.combo_isotope.setVisible(len(isotopes) > 0)

        # Checkbox
        self.check_all = QtWidgets.QCheckBox("Apply range to all elements.")
        self.check_all.setChecked(len(isotopes) == 0)
        self.check_all.setEnabled(len(isotopes) > 0)
        self.check_all.clicked.connect(self.enableComboIsotope)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Minimum:", self.lineedit_min)
        layout_form.addRow("Maximum:", self.lineedit_max)

        self.layout_main.addLayout(layout_form)
        self.layout_main.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignRight)
        self.layout_main.addWidget(self.check_all)

        self.updateLineEdits()

    def enableComboIsotope(self, enabled: bool) -> None:
        self.combo_isotope.setEnabled(not enabled)
        self.updateLineEdits()

    def updateLineEdits(self) -> None:
        self.lineedit_min.setPlaceholderText(str(self.default_range[0]))
        self.lineedit_max.setPlaceholderText(str(self.default_range[1]))
        tmin, tmax = "", ""

        # If the combobox is disabled then shown default range as true text
        if self.combo_isotope.isEnabled():
            # tmin, tmax = "", ""
            # If there is a current isotope then update text to it's value, if exists
            current_isotope = self.combo_isotope.currentText()
            if current_isotope in self.ranges:
                range = self.ranges[current_isotope]
                tmin, tmax = str(range[0]), str(range[1])

        else:
            tmin, tmax = str(self.default_range[0]), str(self.default_range[1])

        self.lineedit_min.setText(tmin)
        self.lineedit_max.setText(tmax)

    def comboChanged(self) -> None:
        self.updateRange(self.previous_isotope)
        self.updateLineEdits()
        self.previous_isotope = self.combo_isotope.currentText()

    def updateRange(self, isotope: str = None) -> None:
        tmin, tmax = self.lineedit_min.text(), self.lineedit_max.text()
        vmin, vmax = self.ranges.get(isotope, self.default_range)

        if tmin != "":
            vmin = tmin if "%" in tmin else float(tmin)
        if tmax != "":
            vmax = tmax if "%" in tmax else float(tmax)

        # Unless at least one value is set return
        if tmin == "" and tmax == "":
            return  # pragma: no cover

        if isotope is not None:
            self.ranges[isotope] = (vmin, vmax)
        else:
            self.ranges = {}
            self.default_range = (vmin, vmax)

    def apply(self) -> None:
        current_isotope = self.combo_isotope.currentText()
        self.updateRange(current_isotope if self.combo_isotope.isEnabled() else None)


class ColocalisationDialog(QtWidgets.QDialog):
    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray = None,
        colors: List[Tuple[float, ...]] = None,
        parent: QtWidgets.QWidget = None,
    ):
        assert data.dtype.names is not None
        super().__init__(parent)
        self.setWindowTitle("Colocalisation")
        self.data = data
        self.mask = mask

        # if colors is None:
        #     colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        # self.cmap = LinearSegmentedColormap.from_list("colocal_cmap", colors)

        self.chart = ColocalisationChart()

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
        x = self.data[n1]
        y = self.data[n2]

        if self.mask is not None:
            x, y = x[self.mask], y[self.mask]

        x, y = normalise(x), normalise(y)

        # Pearson
        r = colocal.pearsonr(x, y)

        # Li
        icq = colocal.li_icq(x, y)

        x, y = x.ravel(), y.ravel()
        if x.size > 10000:  # pragma: no cover
            n = np.random.choice(x.size, 10000)
            x, y = x[n], y[n]

        # Choose a more approriate threshold?
        # TODO this is really slow, python loops?
        t1, a, b = colocal.costes_threshold(x, y)
        t2 = a * t1 + b
        m1, m2 = colocal.manders(
            x, y, t2, t1
        )  # Pass thresholds backwards as per Costes

        self.label_r.setText(f"{r:.2f}")
        self.label_p.setText("")
        self.label_icq.setText(f"{icq:.2f}")
        self.label_m1.setText(f"{m1:.2f}")
        self.label_m2.setText(f"{m2:.2f}")

        self.button_p.setEnabled(True)

        self.chart.drawPoints(x, y)
        self.chart.drawLine(a, b)
        self.chart.drawThresholds(t1, t2)

        self.chart.xaxis.setTitleText(n1)
        self.chart.yaxis.setTitleText(n2)

    def calculatePearsonsProbablity(self) -> None:
        x = self.data[self.combo_name1.currentText()]
        y = self.data[self.combo_name2.currentText()]

        _r, p = colocal.pearsonr_probablity(x, y, mask=self.mask, n=500)
        self.label_p.setText(f"{p:.2f}")

        self.button_p.setEnabled(False)


class ConfigDialog(ApplyDialog):
    configSelected = QtCore.Signal(Config)
    configApplyAll = QtCore.Signal(Config)

    def __init__(self, config: Config, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.config = copy.copy(config)

        self.action_copy = qAction(
            "edit-paste",
            "Copy to Clipboard",
            "Copy the current configuration to the system clipboard.",
            self.copy,
        )
        self.button_copy = qToolButton(action=self.action_copy)

        # Line edits
        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setText(str(self.config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidator(0, 1e9, 4))
        self.lineedit_spotsize.textChanged.connect(self.completeChanged)
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setText(str(self.config.speed))
        self.lineedit_speed.setValidator(DecimalValidator(0, 1e9, 4))
        self.lineedit_speed.textChanged.connect(self.completeChanged)
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setText(str(self.config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidator(0, 1e9, 4))
        self.lineedit_scantime.textChanged.connect(self.completeChanged)

        if isinstance(config, SRRConfig):
            self.lineedit_warmup = QtWidgets.QLineEdit()
            self.lineedit_warmup.setText(str(self.config.warmup))
            self.lineedit_warmup.setValidator(DecimalValidator(0, 1e3, 1))
            self.lineedit_warmup.textChanged.connect(self.completeChanged)
            self.spinbox_offsets = QtWidgets.QSpinBox()
            self.spinbox_offsets.setRange(2, 10)
            self.spinbox_offsets.setValue(self.config._subpixel_size)

        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")

        # Form layout for line edits
        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Spotsize (μm):", self.lineedit_spotsize)
        layout_form.addRow("Speed (μm):", self.lineedit_speed)
        layout_form.addRow("Scantime (s):", self.lineedit_scantime)
        if isinstance(config, SRRConfig):
            layout_form.addRow("Warmup (s):", self.lineedit_warmup)
            layout_form.addRow("Subpixel width:", self.spinbox_offsets)

        layout_options = QtWidgets.QHBoxLayout()
        layout_options.addWidget(self.check_all, 1)
        layout_options.addWidget(self.button_copy, 0, QtCore.Qt.AlignRight)

        self.layout_main.addLayout(layout_form)
        self.layout_main.addLayout(layout_options)

    def updateConfig(self) -> None:
        self.config.spotsize = float(self.lineedit_spotsize.text())
        self.config.speed = float(self.lineedit_speed.text())
        self.config.scantime = float(self.lineedit_scantime.text())
        if isinstance(self.config, SRRConfig):
            self.config.warmup = float(self.lineedit_warmup.text())
            self.config.set_equal_subpixel_offsets(self.spinbox_offsets.value())

    def apply(self) -> None:
        self.updateConfig()
        if self.check_all.isChecked():
            self.configApplyAll.emit(self.config)
        else:
            self.configSelected.emit(self.config)

    def copy(self) -> None:
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
        with io.BytesIO() as fp:
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


class NameEditDialog(QtWidgets.QDialog):
    originalNameRole = QtCore.Qt.UserRole + 1
    namesSelected = QtCore.Signal(dict)

    def __init__(
        self,
        names: List[str],
        allow_remove: bool = False,
        parent: QtWidgets.QWidget = None,
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

    def addNames(self, names: List[str]) -> None:
        for name in names:
            self.addName(name)


class SelectionDialog(ApplyDialog):
    maskSelected = QtCore.Signal(np.ndarray, "QStringList")

    METHODS = {
        "Manual": (None, None),
        "Mean": (np.mean, None),
        "Median": (np.median, None),
        "Otsu": (otsu, None),
        "K-means": (kmeans.thresholds, ("k: ", 3, (2, 9))),
    }
    COMPARISION = {">": np.greater, "<": np.less, "=": np.equal}

    def __init__(self, graphics: LaserGraphicsView, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Selection")

        self.graphics = graphics

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
            "Interscet selection with current selection."
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
        data = self.graphics.data
        if self.check_limit_threshold.isChecked() and self.graphics.mask is not None:
            data = data[self.graphics.mask]

        # Remove nans
        data = data[~np.isnan(data)]

        # Enable lineedit if manual mode
        self.lineedit_manual.setEnabled(method == "Manual")

        op, var = SelectionDialog.METHODS[method]

        # Compute new threshold
        if method == "Manual":
            self.lineedit_manual.setEnabled(True)
            self.spinbox_method.setEnabled(False)
            self.spinbox_comparison.setEnabled(False)
            self.threshold = float(self.lineedit_manual.text())
        else:
            self.lineedit_manual.setEnabled(False)
            if var is not None:
                self.spinbox_method.setEnabled(True)
                self.spinbox_method.setPrefix(var[0])
                self.spinbox_method.setRange(*var[2])
                self.spinbox_method.setValue(var[1])
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
        mask = comparison(self.graphics.data, self.threshold)
        state = "intersect" if self.check_limit_selection.isChecked() else None
        self.maskSelected.emit(mask, [state])
        self.refresh()


class StatsDialog(QtWidgets.QDialog):
    isotope_changed = QtCore.Signal(str)

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        units: Dict[str, str],
        isotope: str,
        pixel_size: Tuple[float, float] = None,
        colorranges: dict = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Statistics")

        self.data = self.prepareData(data, mask)
        self.units = units
        self.pixel_size = pixel_size
        self.colorranges = colorranges

        self.chart = HistogramChart()

        self.button_clipboard = QtWidgets.QPushButton("Copy to Clipboard")
        self.button_clipboard.pressed.connect(self.copyToClipboard)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.close)
        self.button_box.addButton(
            self.button_clipboard, QtWidgets.QDialogButtonBox.ActionRole
        )

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.addItems(self.data.dtype.names or [isotope])
        self.combo_isotope.setCurrentText(isotope)
        self.combo_isotope.currentIndexChanged.connect(self.updateStats)
        self.combo_isotope.currentTextChanged.connect(self.isotope_changed)

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
        layout.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignRight)
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

        data += "<tr><td>Name</td><td>Unit</td><td>Min</td><td>Max</td><td>Mean</td>"
        "<td>Median</td><td>Std</tr>"
        text += "Name\tUnit\tMin\tMax\tMean\tMedian\tStd\n"

        for name in self.data.dtype.names:
            nd = self.data[name]
            unit = self.units.get(name, "")
            nd = nd[~np.isnan(nd)]

            data += f"<tr><td>{name}</td><td>{unit}</td><td>{np.min(nd)}</td>"
            f"<td>{np.max(nd)}</td><td>{np.mean(nd)}</td><td>{np.median(nd)}</td>"
            f"<td>{np.std(nd)}</td></tr>"

            text += f"{name}\t{unit}\t{np.min(nd)}\t{np.max(nd)}\t"
            f"{np.mean(nd)}\t{np.median(nd)}\t{np.std(nd)}\n"

        text = text.rstrip("\n")
        data += "</table>"

        mime = QtCore.QMimeData()
        mime.setHtml(data)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def updateStats(self) -> None:
        isotope = self.combo_isotope.currentText()
        data = self.data[isotope]
        unit = self.units.get(isotope, "")

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

        self.chart.setHistogram(data)

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
