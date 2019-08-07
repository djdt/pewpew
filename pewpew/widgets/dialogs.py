import copy
import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.text import Text

from laserlib import LaserCalibration, LaserConfig
from laserlib.krisskross import KrissKrossConfig

from pewpew.widgets.canvases import BasicCanvas
from pewpew.validators import (
    DecimalValidator,
    DecimalValidatorNoZero,
    PercentOrDecimalValidator,
)

from typing import Dict, List, Tuple, Union


class ApplyDialog(QtWidgets.QDialog):

    applyPressed = QtCore.Signal(QtCore.QObject)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.layout_form = QtWidgets.QFormLayout()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply,
            self,
        )
        self.button_box.clicked.connect(self.buttonClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_form)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def buttonClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.Apply:
            if self.complete():
                self.apply()
                self.applyPressed.emit(self)
            else:
                self.error()
        elif sb == QtWidgets.QDialogButtonBox.Ok:
            if self.complete():
                self.apply()
                self.applyPressed.emit(self)
                self.accept()
            else:
                self.error()
        else:
            self.reject()

    def apply(self) -> None:
        pass

    def complete(self) -> bool:
        return True

    def error(self) -> None:
        pass


class CalibrationDialog(ApplyDialog):
    def __init__(
        self,
        calibrations: Dict[str, LaserCalibration],
        current_isotope: str,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.calibrations = copy.deepcopy(calibrations)

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
        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.addItems(list(self.calibrations.keys()))
        self.combo_isotopes.setCurrentText(current_isotope)
        self.previous_index = self.combo_isotopes.currentIndex()
        self.combo_isotopes.currentIndexChanged.connect(self.comboChanged)

        # Check all
        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")

        # Button to plot
        self.button_plot = QtWidgets.QPushButton("Plot")
        self.button_plot.setEnabled(
            self.calibrations[current_isotope].points is not None
        )
        self.button_plot.pressed.connect(self.showCurve)

        layout_isotopes = QtWidgets.QHBoxLayout()
        layout_isotopes.addWidget(self.button_plot, 0, QtCore.Qt.AlignLeft)
        layout_isotopes.addWidget(self.combo_isotopes, 0, QtCore.Qt.AlignRight)

        # Form layout for line edits
        self.layout_form.addRow("Gradient:", self.lineedit_gradient)
        self.layout_form.addRow("Intercept:", self.lineedit_intercept)
        self.layout_form.addRow("Unit:", self.lineedit_unit)
        self.layout().insertLayout(1, layout_isotopes)
        # self.layout().insertLayout(1, self.combo_isotopes, 1, QtCore.Qt.AlignRight)
        self.layout().insertWidget(2, self.check_all)

        self.updateLineEdits()

    def updateLineEdits(self) -> None:
        name = self.combo_isotopes.currentText()

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
        if unit is None:
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
        previous = self.combo_isotopes.itemText(self.previous_index)
        self.updateCalibration(previous)
        self.updateLineEdits()
        self.previous_index = self.combo_isotopes.currentIndex()
        self.button_plot.setEnabled(
            self.calibrations[self.combo_isotopes.currentText()].points is not None
        )

    def showCurve(self) -> None:
        dlg = CalibrationCurveDialog(
            self.calibrations[self.combo_isotopes.currentText()], parent=self
        )
        dlg.show()

    def apply(self) -> None:
        self.updateCalibration(self.combo_isotopes.currentText())


class CalibrationCurveDialog(QtWidgets.QDialog):
    def __init__(self, calibration: LaserCalibration, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Curve")
        self.canvas = BasicCanvas(parent=self)
        ax = self.canvas.figure.subplots()

        x = calibration.concentrations()
        y = calibration.counts()
        x0, x1 = 0.0, np.nanmax(x) * 1.1

        m = calibration.gradient
        b = calibration.intercept

        xlabel = "Concentration"
        if calibration.unit != "":
            xlabel += f" ({calibration.unit})"

        ax.scatter(x, y, color="black")
        ax.plot([x0, x1], [m * x0 + b, m * x1 + b], ls=":", lw=1.5, color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")

        text = Text(
            x=0.05,
            y=0.95,
            text=str(calibration),
            transform=ax.transAxes,
            color="black",
            fontsize=12,
            horizontalalignment="left",
            verticalalignment="top",
        )

        ax.add_artist(text)

        self.canvas.draw()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        context_menu = QtWidgets.QMenu(self)
        action_copy = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy Image", self
        )
        action_copy.setStatusTip("Copy image to clipboard.")
        action_copy.triggered.connect(self.canvas.copyToClipboard)
        context_menu.addAction(action_copy)
        context_menu.popup(event.globalPos())


class ColorRangeDialog(ApplyDialog):
    def __init__(
        self,
        current_range: Tuple[Union[float, str], Union[float, str]],
        parent: QtWidgets.QWidget = None,
    ):
        self.range = current_range
        super().__init__(parent)
        self.setWindowTitle("Colormap Range")

        self.lineedit_min = QtWidgets.QLineEdit()
        self.lineedit_min.setPlaceholderText(str(current_range[0]))
        self.lineedit_min.setToolTip("Percentile for minium colormap value.")
        self.lineedit_min.setValidator(
            PercentOrDecimalValidator(-1e99, 1e99, parent=self.lineedit_min)
        )
        self.lineedit_max = QtWidgets.QLineEdit()
        self.lineedit_max.setPlaceholderText(str(current_range[1]))
        self.lineedit_max.setValidator(
            PercentOrDecimalValidator(-1e99, 1e99, parent=self.lineedit_max)
        )
        self.lineedit_max.setToolTip("Percentile for maximum colormap value.")

        self.layout_form.addRow("Minimum:", self.lineedit_min)
        self.layout_form.addRow("Maximum:", self.lineedit_max)

    def updateRange(self) -> None:
        min, max = self.lineedit_min.text(), self.lineedit_max.text()
        if min == "":
            min = self.range[0]
        elif "%" not in min:
            min = float(min)
        if max == "":
            max = self.range[1]
        elif "%" not in max:
            max = float(max)
        self.range = (min, max)

    def apply(self) -> None:
        self.updateRange()


class ConfigDialog(ApplyDialog):
    def __init__(self, config: LaserConfig, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.config = copy.copy(config)

        # Line edits
        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setPlaceholderText(str(self.config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidator(0, 1e3, 0))
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setPlaceholderText(str(self.config.speed))
        self.lineedit_speed.setValidator(DecimalValidator(0, 1e3, 0))
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setPlaceholderText(str(self.config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidator(0, 1e3, 4))

        if isinstance(config, KrissKrossConfig):
            self.lineedit_warmup = QtWidgets.QLineEdit()
            self.lineedit_warmup.setPlaceholderText(
                str(self.config.warmup)  # type: ignore
            )
            self.lineedit_warmup.setValidator(DecimalValidator(0, 100, 1))
            self.spinbox_offsets = QtWidgets.QSpinBox()
            self.spinbox_offsets.setRange(2, 10)
            self.spinbox_offsets.setValue(self.config._subpixel_size)  # type: ignore

        # Form layout for line edits
        self.layout_form.addRow("Spotsize (μm):", self.lineedit_spotsize)
        self.layout_form.addRow("Speed (μm):", self.lineedit_speed)
        self.layout_form.addRow("Scantime (s):", self.lineedit_scantime)

        if isinstance(config, KrissKrossConfig):
            self.layout_form.addRow("Warmup (s):", self.lineedit_warmup)
            self.layout_form.addRow("Subpixel width:", self.spinbox_offsets)

        # Checkbox
        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")
        self.layout().insertWidget(1, self.check_all)

    def updateConfig(self) -> None:
        if self.lineedit_spotsize.text() != "":
            self.config.spotsize = float(self.lineedit_spotsize.text())
        if self.lineedit_speed.text() != "":
            self.config.speed = float(self.lineedit_speed.text())
        if self.lineedit_scantime.text() != "":
            self.config.scantime = float(self.lineedit_scantime.text())
        if isinstance(self.config, KrissKrossConfig):
            if self.lineedit_warmup.text() != "":
                self.config.warmup = float(self.lineedit_warmup.text())
            self.config.set_equal_subpixel_offsets(self.spinbox_offsets.value())

    def apply(self) -> None:
        self.updateConfig()


class MultipleDirDialog(QtWidgets.QFileDialog):
    def __init__(self, title: str, directory: str, parent: QtWidgets.QWidget = None):
        super().__init__(parent, title, directory)
        self.setFileMode(QtWidgets.QFileDialog.Directory)
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        for view in self.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
            if isinstance(view.model(), QtWidgets.QFileSystemModel):
                view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    @staticmethod
    def getExistingDirectories(
        parent: QtWidgets.QWidget, title: str, directory: str
    ) -> List[str]:
        dlg = MultipleDirDialog(title, directory, parent)
        if dlg.exec():
            return list(dlg.selectedFiles())
        else:
            return []


class StatsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        data: np.ndarray,
        range: Tuple[Union[str, float], Union[str, float]],
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Statistics")

        self.canvas = BasicCanvas(figsize=(6, 2))
        self.canvas.ax = self.canvas.figure.add_subplot(111)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.close)

        stats_left = QtWidgets.QFormLayout()
        stats_left.addRow("Shape:", QtWidgets.QLabel(str(data.shape)))
        stats_left.addRow("Size:", QtWidgets.QLabel(str(data.size)))

        # Ensure no nans
        data = data[~np.isnan(data)]
        stats_right = QtWidgets.QFormLayout()
        stats_right.addRow("Min:", QtWidgets.QLabel(f"{np.min(data):.4g}"))
        stats_right.addRow("Max:", QtWidgets.QLabel(f"{np.max(data):.4g}"))
        stats_right.addRow("Mean:", QtWidgets.QLabel(f"{np.mean(data):.4g}"))
        stats_right.addRow("Median:", QtWidgets.QLabel(f"{np.median(data):.4g}"))

        stats_box = QtWidgets.QGroupBox()
        stats_layout = QtWidgets.QHBoxLayout()
        stats_layout.addLayout(stats_left)
        stats_layout.addLayout(stats_right)
        stats_box.setLayout(stats_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(stats_box)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        # Calculate the range
        if isinstance(range[0], str):
            vmin = np.percentile(data, float(range[0].rstrip("%")))
        else:
            vmin = float(range[0])
        if isinstance(range[1], str):
            vmax = np.percentile(data, float(range[1].rstrip("%")))
        else:
            vmax = float(range[1])

        plot_data = data[np.logical_and(data >= vmin, data <= vmax)]
        self.plot(plot_data)

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        action_copy_image = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy Image", self
        )
        action_copy_image.setStatusTip("Copy image to clipboard.")
        action_copy_image.triggered.connect(self.canvas.copyToClipboard)

        context_menu = QtWidgets.QMenu(self)
        context_menu.addAction(action_copy_image)
        context_menu.popup(event.globalPos())

    def plot(self, data: np.ndarray) -> None:
        highlight = self.palette().color(QtGui.QPalette.Highlight).name()
        self.canvas.ax.hist(data.ravel(), bins="auto", color=highlight)
        self.canvas.draw()
