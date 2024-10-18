import logging
from importlib.metadata import version
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
import pyqtgraph
from pewlib.config import SpotConfig
from pewlib.laser import Laser
from pewlib.process import peakfinding
from pewlib.process.calc import view_as_blocks
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.charts.signal import SignalView
from pewpew.graphics.colortable import get_table
from pewpew.graphics.imageitems import ScaledImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions
from pewpew.validators import DecimalValidator, DecimalValidatorNoZero, OddIntValidator
from pewpew.widgets.dialogs import NameEditDialog
from pewpew.widgets.wizards.import_ import FormatPage
from pewpew.widgets.wizards.options import PathAndOptionsPage

logger = logging.getLogger(__name__)


class SpotImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_agilent = 1
    page_csv = 2
    page_numpy = 3
    page_perkinelmer = 4
    page_text = 5
    page_thermo = 6
    page_spot_peaks = 7
    page_spot_image = 8
    page_spot_config = 9

    laserImported = QtCore.Signal(Path, Laser)

    def __init__(
        self,
        paths: list[str] | list[Path] = [],
        config: SpotConfig | None = None,
        options: GraphicsOptions | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spot Import Wizard")

        paths = [Path(p) for p in paths]

        config = config or SpotConfig()

        overview = (
            "This wizard is for importing spotwise collected data."
            "To begin select the format of the file(s) being imported."
        )

        format_page = FormatPage(
            overview,
            page_id_dict={
                "agilent": self.page_agilent,
                "csv": self.page_csv,
                "numpy": self.page_numpy,
                "perkinelmer": self.page_perkinelmer,
                "text": self.page_text,
                "thermo": self.page_thermo,
            },
            parent=self,
        )

        self.setPage(self.page_format, format_page)
        self.setPage(
            self.page_agilent,
            PathAndOptionsPage(
                paths,
                "agilent",
                nextid=self.page_spot_peaks,
                multiplepaths=True,
                register_laser_fields=True,
                parent=self,
            ),
        )
        self.setPage(
            self.page_csv,
            PathAndOptionsPage(
                paths,
                "csv",
                nextid=self.page_spot_peaks,
                multiplepaths=True,
                parent=self,
            ),
        )
        self.setPage(
            self.page_numpy,
            PathAndOptionsPage(
                paths,
                "numpy",
                nextid=self.page_spot_peaks,
                multiplepaths=True,
                parent=self,
            ),
        )
        self.setPage(
            self.page_perkinelmer,
            PathAndOptionsPage(
                paths,
                "perkinelmer",
                nextid=self.page_spot_peaks,
                multiplepaths=True,
                parent=self,
            ),
        )
        self.setPage(
            self.page_text,
            PathAndOptionsPage(
                paths,
                "text",
                nextid=self.page_spot_peaks,
                multiplepaths=True,
                parent=self,
            ),
        )
        self.setPage(
            self.page_thermo,
            PathAndOptionsPage(
                paths,
                "thermo",
                nextid=self.page_spot_peaks,
                multiplepaths=True,
                parent=self,
            ),
        )

        self.setPage(self.page_spot_peaks, SpotPeaksPage(parent=self))
        self.setPage(self.page_spot_image, SpotImagePage(options, parent=self))
        self.setPage(self.page_spot_config, SpotConfigPage(config, parent=self))

    def accept(self) -> None:
        peaks = self.field("peaks")

        x = int(self.field("shape_x"))
        y = int(self.field("shape_y"))

        data = np.full(
            (y, x), np.nan, dtype=[(name, np.float64) for name in peaks.dtype.names]
        )
        for name in data.dtype.names:
            data[name].flat = peaks[name][self.field("integration")]

        if self.field("raster"):
            data[::2, :] = data[::2, ::-1]

        config = SpotConfig(
            float(self.field("spotsize")), float(self.field("spotsize_y"))
        )

        if self.field("agilent"):
            paths = [Path(p) for p in self.field("agilent.paths")]
        elif self.field("csv"):
            paths = [Path(p) for p in self.field("csv.paths")]
        elif self.field("numpy"):
            paths = [Path(p) for p in self.field("numpy.paths")]
        elif self.field("perkinelmer"):
            paths = [Path(p) for p in self.field("perkinelmer.paths")]
        elif self.field("text"):
            paths = [Path(p) for p in self.field("text.paths")]
        elif self.field("thermo"):
            paths = [Path(p) for p in self.field("thermo.paths")]
        else:  # pragma: no cover
            raise ValueError("Invalid filetype selection.")

        info = self.field("laserinfo")[0]
        self.laserImported.emit(paths[0], Laser(data, config=config, info=info))
        super().accept()


class SpotPeakOptions(QtWidgets.QGroupBox):
    optionsChanged = QtCore.Signal()

    def __init__(self, name: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(name, parent=parent)
        self.setLayout(QtWidgets.QFormLayout())

    def args(self) -> dict:
        raise NotImplementedError

    def isComplete(self) -> bool:
        return True


class ConstantPeakOptions(SpotPeakOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Constant Options", parent)

        self.lineedit_minimum = QtWidgets.QLineEdit("0.0")
        self.lineedit_minimum.setValidator(DecimalValidator(-1e99, 1e99, 4))
        self.lineedit_minimum.textChanged.connect(self.optionsChanged)

        self.layout().addRow("Minimum value:", self.lineedit_minimum)

    def args(self) -> dict:
        return {
            "minimum": float(self.lineedit_minimum.text()),
        }

    def isComplete(self) -> bool:
        return self.lineedit_minimum.hasAcceptableInput()


class CWTPeakOptions(SpotPeakOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("CWT Options", parent)
        self.lineedit_minwidth = QtWidgets.QLineEdit("3")
        self.lineedit_minwidth.setValidator(QtGui.QIntValidator(1, 992))
        self.lineedit_minwidth.textChanged.connect(self.optionsChanged)
        self.lineedit_maxwidth = QtWidgets.QLineEdit("9")
        self.lineedit_maxwidth.setValidator(QtGui.QIntValidator(2, 100))
        self.lineedit_maxwidth.textChanged.connect(self.optionsChanged)

        self.lineedit_minlen = QtWidgets.QLineEdit("3")
        self.lineedit_minlen.setValidator(QtGui.QIntValidator(1, 20))
        self.lineedit_minlen.textChanged.connect(self.optionsChanged)
        self.lineedit_minsnr = QtWidgets.QLineEdit("3.3")
        self.lineedit_minsnr.setValidator(DecimalValidatorNoZero(0, 100, 1))
        self.lineedit_minsnr.textChanged.connect(self.optionsChanged)
        self.lineedit_width_factor = QtWidgets.QLineEdit("2.5")
        self.lineedit_width_factor.setValidator(DecimalValidatorNoZero(0, 10, 1))
        self.lineedit_width_factor.textChanged.connect(self.optionsChanged)

        self.layout().addRow("Minimum width:", self.lineedit_minwidth)
        self.layout().addRow("Maximum width:", self.lineedit_maxwidth)
        self.layout().addRow("Width factor:", self.lineedit_width_factor)
        self.layout().addRow("Minimum ridge SNR:", self.lineedit_minsnr)
        self.layout().addRow("Minimum ridge length:", self.lineedit_minlen)

    def args(self) -> dict:
        return {
            "width": (
                int(self.lineedit_minwidth.text()),
                int(self.lineedit_maxwidth.text()),
            ),
            "snr": float(self.lineedit_minsnr.text()),
            "length": int(self.lineedit_minlen.text()),
            "width_factor": float(self.lineedit_width_factor.text()),
        }

    def isComplete(self) -> bool:
        if not all(
            x.hasAcceptableInput()
            for x in [
                self.lineedit_minwidth,
                self.lineedit_maxwidth,
                self.lineedit_minsnr,
                self.lineedit_minlen,
                self.lineedit_width_factor,
            ]
        ):
            return False
        return True


class WindowedPeakOptions(SpotPeakOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Window Options", parent)

        self.lineedit_window_size = QtWidgets.QLineEdit("9")
        self.lineedit_window_size.setValidator(OddIntValidator(3, 999))
        self.lineedit_window_size.textChanged.connect(self.optionsChanged)

        self.combo_baseline_method = QtWidgets.QComboBox()
        self.combo_baseline_method.addItems(["Mean", "Median"])
        self.combo_baseline_method.currentTextChanged.connect(self.optionsChanged)

        self.combo_thresh_method = QtWidgets.QComboBox()
        self.combo_thresh_method.addItems(["Constant", "Std"])
        self.combo_thresh_method.setCurrentText("Std")
        self.combo_thresh_method.currentTextChanged.connect(self.optionsChanged)

        self.lineedit_sigma = QtWidgets.QLineEdit("3.0")
        self.lineedit_sigma.setValidator(DecimalValidator(-1e99, 1e99, 4))
        self.lineedit_sigma.textChanged.connect(self.optionsChanged)

        self.layout().addRow("Window size:", self.lineedit_window_size)
        self.layout().addRow("Window baseline:", self.combo_baseline_method)
        self.layout().addRow("Window threshold:", self.combo_thresh_method)
        self.layout().addRow("Threshold:", self.lineedit_sigma)

    def args(self) -> dict:
        return {
            "method": self.combo_baseline_method.currentText(),
            "thresh": self.combo_thresh_method.currentText(),
            "size": int(self.lineedit_window_size.text()),
            "value": float(self.lineedit_sigma.text()),
        }

    def isComplete(self) -> bool:
        if (
            not self.lineedit_window_size.hasAcceptableInput()
            or not self.lineedit_sigma.hasAcceptableInput()
        ):
            return False
        return True


class SpotPeaksPage(QtWidgets.QWizardPage):
    peaksChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Spot Peak Detection")

        self._datas: list[np.ndarray] = []
        self._infos: list[dict[str, str]] = []
        self.peaks: np.ndarray | None = None
        self.options = {
            "Constant": ConstantPeakOptions(self),
            "CWT": CWTPeakOptions(self),
            "Moving window": WindowedPeakOptions(self),
        }

        self.check_single_spot = QtWidgets.QCheckBox("One spot per line.")
        self.check_single_spot.clicked.connect(self.completeChanged)

        self.combo_peak_method = QtWidgets.QComboBox()
        self.combo_peak_method.addItems(list(self.options.keys()))

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.currentIndexChanged.connect(self.onElementChanged)

        self.stack = QtWidgets.QStackedWidget()
        for option in self.options.values():
            self.stack.addWidget(option)
            option.optionsChanged.connect(self.completeChanged)
            option.optionsChanged.connect(self.updatePeaks)

        self.combo_peak_method.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.combo_peak_method.currentIndexChanged.connect(self.updatePeaks)

        self.chart = SignalView(parent=self)
        self.series = {
            "signal": self.chart.addLine("Signal", np.array([0])),
            "peaks": self.chart.addScatterSeries(
                "Peaks",
                np.array([0]),
                np.array([0]),
                brush=QtGui.QBrush(QtCore.Qt.GlobalColor.red),
            ),
            "lefts": self.chart.addScatterSeries(
                "Lefts",
                np.array([0]),
                np.array([0]),
                brush=QtGui.QBrush(QtCore.Qt.GlobalColor.darkBlue),
            ),
            "rights": self.chart.addScatterSeries(
                "Rights",
                np.array([0]),
                np.array([0]),
                brush=QtGui.QBrush(QtCore.Qt.GlobalColor.darkGreen),
            ),
        }

        self.combo_base_method = QtWidgets.QComboBox()
        self.combo_base_method.addItems(
            ["baseline", "edge", "prominence", "minima", "zero"]
        )
        self.combo_base_method.currentTextChanged.connect(self.updatePeaks)
        # self.combo_base_method.currentTextChanged.connect(self.drawThresholds)
        self.combo_height_method = QtWidgets.QComboBox()
        self.combo_height_method.addItems(["center", "maxima"])
        self.combo_height_method.setCurrentText("maxima")
        self.combo_height_method.currentTextChanged.connect(self.updatePeaks)

        self.lineedit_minarea = QtWidgets.QLineEdit("0")
        self.lineedit_minarea.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minarea.textChanged.connect(self.completeChanged)
        self.lineedit_minarea.textChanged.connect(self.updatePeaks)
        self.lineedit_minheight = QtWidgets.QLineEdit("0")
        self.lineedit_minheight.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minheight.textChanged.connect(self.completeChanged)
        self.lineedit_minheight.textChanged.connect(self.updatePeaks)
        self.lineedit_minwidth = QtWidgets.QLineEdit("0")
        self.lineedit_minwidth.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minwidth.textChanged.connect(self.completeChanged)
        self.lineedit_minwidth.textChanged.connect(self.updatePeaks)

        self.lineedit_count = QtWidgets.QLineEdit()
        self.lineedit_count.setReadOnly(True)

        layout_form_controls = QtWidgets.QFormLayout()
        layout_form_controls.addRow("Peak base:", self.combo_base_method)
        layout_form_controls.addRow("Peak height:", self.combo_height_method)
        layout_form_controls.addRow("Minimum area:", self.lineedit_minarea)
        layout_form_controls.addRow("Minimum height:", self.lineedit_minheight)
        layout_form_controls.addRow("Minimum width:", self.lineedit_minwidth)
        layout_form_controls.addRow("Peak count:", self.lineedit_count)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addWidget(self.check_single_spot)
        layout_controls.addWidget(self.combo_peak_method)
        layout_controls.addWidget(self.stack)
        layout_controls.addLayout(layout_form_controls)

        layout_chart = QtWidgets.QVBoxLayout()
        layout_chart.addWidget(self.chart, 1)
        layout_chart.addWidget(self.combo_element, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_controls)
        layout.addLayout(layout_chart, 1)
        self.setLayout(layout)

        self.registerField(
            "element", self.combo_element, "currentText", "currentTextChanged"
        )
        self.registerField(
            "base_method", self.combo_base_method, "currentText", "currentTextChanged"
        )
        self.registerField(
            "height_method",
            self.combo_height_method,
            "currentText",
            "currentTextChanged",
        )

        self.registerField("peaks", self, "peaks_prop")

    def isComplete(self) -> bool:
        if not self.stack.widget(self.stack.currentIndex()).isComplete():
            return False
        if not all(
            x.hasAcceptableInput()
            for x in [
                self.lineedit_minarea,
                self.lineedit_minheight,
                self.lineedit_minwidth,
            ]
        ):
            return False
        return True

    def getPeaks(self) -> np.ndarray | None:
        return self.peaks

    def setPeaks(self, peaks: np.ndarray) -> None:
        self.peaks = peaks
        self.peaksChanged.emit()

    def initializePage(self) -> None:
        data = np.concatenate([x.flat for x in self.field("laserdata")], axis=0)
        self.combo_element.blockSignals(True)
        self.combo_element.clear()
        self.combo_element.addItems(data.dtype.names)
        self.combo_element.blockSignals(False)

        data = data[self.combo_element.currentText()]
        self.options["Constant"].lineedit_minimum.setText(
            f"{np.percentile(data, 25):.2f}"
        )

        self.drawSignal(data)
        self.updatePeaks()

    def onElementChanged(self) -> None:
        data = np.concatenate([x.flat for x in self.field("laserdata")], axis=0)
        data = data[self.combo_element.currentText()]
        self.drawSignal(data)
        self.updatePeaks()

    def drawSignal(self, data: np.ndarray) -> None:
        self.series["signal"].setData(data)

    def clearPeaks(self) -> None:
        self.peaks = None
        self.lineedit_count.setText("0")
        for name in ["peaks", "lefts", "rights"]:
            self.series[name].setVisible(False)

    def drawPeaks(self, peaks: np.ndarray) -> None:
        self.series["peaks"].setData(peaks["top"], peaks["height"] + peaks["base"])
        self.series["lefts"].setData(peaks["left"], peaks["base"])
        self.series["rights"].setData(peaks["right"], peaks["base"])
        for name in ["peaks", "lefts", "rights"]:
            self.series[name].setVisible(True)

    def clearThresholds(self) -> None:
        return
        for name in list(self.chart.series.keys()):
            if name not in ["signal", "peaks", "lefts", "rights"]:
                self.chart.chart().removeSeries(self.chart.series.pop(name))

    def drawThresholds(self, thresholds: dict) -> None:
        return
        # colors = iter(sequential)
        for name, value in thresholds.items():
            color = next(colors)
            if name in self.chart.series:
                self.chart.setSeries(name, value)
            else:
                self.chart.addLineSeries(name, value, color=color, linewidth=2.0)

    def updatePeaks(self) -> None:
        self.peaksChanged.emit()

        if not self.isComplete():
            self.clearThresholds()
            self.clearPeaks()
            return

        method = self.combo_peak_method.currentText()
        args = self.options[method].args()
        data = np.concatenate([x.flat for x in self.field("laserdata")], axis=0)
        data = data[self.combo_element.currentText()]

        if method == "Constant":
            thresholds = {"baseline": np.full(data.size, args["minimum"])}
            diff = np.diff((data > thresholds["baseline"]).astype(np.int8), prepend=0)
            lefts = np.flatnonzero(diff == 1)
            rights = np.flatnonzero(diff == -1)

        elif method == "CWT":
            thresholds = {}
            windows = np.arange(args["width"][0], args["width"][1])
            cwt_coef = peakfinding.cwt(data, windows, peakfinding.ricker_wavelet)
            ridges = peakfinding._cwt_identify_ridges(
                cwt_coef, windows, gap_threshold=None
            )
            ridges, ridge_maxima = peakfinding._cwt_filter_ridges(
                ridges,
                cwt_coef,
                noise_window=windows[-1] * 4,
                min_length=args["length"],
                min_snr=args["snr"],
            )

            if ridges.size == 0:
                self.clearPeaks()
                return

            widths = (np.take(windows, ridge_maxima[0]) * args["width_factor"]).astype(
                int
            )
            lefts = np.clip(ridge_maxima[1] - widths // 2, 0, data.size - 1)
            rights = np.clip(ridge_maxima[1] + widths // 2, 1, data.size)

        elif method == "Moving window":
            size = args["size"]
            x_pad = np.pad(data, [size // 2, size - size // 2 - 1], mode="edge")
            view = view_as_blocks(x_pad, (size,), (1,))
            if args["method"] == "Mean":
                baseline = np.mean(view, axis=1)
            elif args["method"] == "Median":
                baseline = np.median(view, axis=1)
            else:
                raise ValueError("Method must be 'Mean' or 'Median'.")

            if args["thresh"] == "Std":
                threshold = np.std(view, axis=1) * args["value"]
            elif args["thresh"] == "Constant":
                threshold = np.full(data.size, args["value"])
            else:
                raise ValueError("Threshold must be 'Std' or 'Constant'.")

            thresholds = {"baseline": baseline, "threshold": baseline + threshold}

            diff = np.diff((data > (baseline + threshold)).astype(np.int8), prepend=0)
            lefts = np.flatnonzero(diff == 1)
            rights = np.flatnonzero(diff == -1)
        else:
            raise ValueError("Method must be 'Constant', 'CWT' or 'Moving Window'.")

        self.clearThresholds()
        self.drawThresholds(thresholds)

        if rights.size > lefts.size:
            rights = rights[1:]
        elif lefts.size > rights.size:
            lefts = lefts[:-1]

        if lefts.size == 0 or rights.size == 0 or lefts.size != rights.size:
            self.clearPeaks()
            return

        self.peaks = peakfinding.peaks_from_edges(
            data,
            lefts,
            rights,
            base_method=self.combo_base_method.currentText(),
            height_method=self.combo_height_method.currentText(),
            baseline=thresholds.get("baseline", None),
        )

        self.peaks = peakfinding.filter_peaks(
            self.peaks,
            min_area=float(self.lineedit_minarea.text()),
            min_height=float(self.lineedit_minheight.text()),
            min_width=float(self.lineedit_minwidth.text()),
        )
        assert isinstance(self.peaks, np.ndarray)

        self.drawPeaks(self.peaks)
        self.lineedit_count.setText(f"{self.peaks.size}")

    def validatePage(self) -> bool:
        if self.peaks is None or self.peaks.size == 0:
            return False

        data = np.concatenate([x.flat for x in self.field("laserdata")], axis=0)
        peaks = self.field("peaks")

        peakdata = np.empty(
            peaks.size,
            dtype=[(name, peakfinding.PEAK_DTYPE) for name in data.dtype.names],
        )
        for name in data.dtype.names:
            if name == self.field("element"):
                peakdata[name] = peaks
            else:
                peakdata[name] = peakfinding.peaks_from_edges(
                    data[name],
                    peaks["left"],
                    peaks["right"],
                    self.field("base_method"),
                    self.field("height_method"),
                )
        self.peaks = peakdata
        return True

    peaks_prop = QtCore.Property("QVariant", getPeaks, setPeaks, notify=peaksChanged)


class SpotImagePage(QtWidgets.QWizardPage):
    def __init__(
        self,
        options: GraphicsOptions | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spot Image Preview")

        if options is None:
            options = GraphicsOptions()
        self.image: ScaledImageItem | None = None

        self.lineedit_shape_x = QtWidgets.QLineEdit("0")
        self.lineedit_shape_x.setValidator(QtGui.QIntValidator(1, 99999))
        self.lineedit_shape_x.textChanged.connect(self.updateImage)

        self.lineedit_shape_y = QtWidgets.QLineEdit("0")
        self.lineedit_shape_y.setValidator(QtGui.QIntValidator(1, 99999))
        self.lineedit_shape_y.textChanged.connect(self.updateImage)

        self.lineedit_count = QtWidgets.QLineEdit()
        self.lineedit_count.setReadOnly(True)
        self.lineedit_diff = QtWidgets.QLineEdit()
        self.lineedit_diff.setReadOnly(True)

        self.combo_integ = QtWidgets.QComboBox()
        self.combo_integ.addItems(["area", "height"])
        self.combo_integ.currentTextChanged.connect(self.updateImage)

        self.check_raster = QtWidgets.QCheckBox("Alternate line raster direction.")
        self.check_raster.toggled.connect(self.updateImage)

        self.graphics = LaserGraphicsView(options, parent=self)

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.currentIndexChanged.connect(self.updateImage)

        layout_shape = QtWidgets.QHBoxLayout()
        layout_shape.addWidget(self.lineedit_shape_x, 1)
        layout_shape.addWidget(QtWidgets.QLabel("x"), 0)
        layout_shape.addWidget(self.lineedit_shape_y, 1)

        layout_form_controls = QtWidgets.QFormLayout()
        layout_form_controls.addRow("Shape X:", self.lineedit_shape_x)
        layout_form_controls.addRow("Shape Y:", self.lineedit_shape_y)
        layout_form_controls.addRow(self.check_raster)
        layout_form_controls.addRow("Peak count:", self.lineedit_count)
        layout_form_controls.addRow("Difference:", self.lineedit_diff)
        layout_form_controls.addRow("Use peak:", self.combo_integ)

        controls = QtWidgets.QGroupBox("Controls")
        controls.setLayout(layout_form_controls)

        layout_chart = QtWidgets.QVBoxLayout()
        layout_chart.addWidget(self.graphics, 1)
        layout_chart.addWidget(self.combo_element, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(controls, 0)
        layout.addLayout(layout_chart, 1)
        self.setLayout(layout)

        self.registerField("shape_x", self.lineedit_shape_x)
        self.registerField("shape_y", self.lineedit_shape_y)
        self.registerField("raster", self.check_raster)
        self.registerField(
            "integration", self.combo_integ, "currentText", "currentTextChanged"
        )

    def initializePage(self) -> None:
        data = np.concatenate([x.flat for x in self.field("laserdata")], axis=0)
        self.combo_element.blockSignals(True)
        self.combo_element.clear()
        self.combo_element.addItems(data.dtype.names)
        self.combo_element.setCurrentText(self.field("element"))
        self.combo_element.blockSignals(False)

        peaks = self.field("peaks")
        x = int(np.sqrt(peaks.size))
        y = int(peaks.size / x)

        if self.lineedit_shape_x.text() == "0":
            self.lineedit_shape_x.setText(f"{x}")
            self.lineedit_shape_y.setText(f"{y}")

        self.lineedit_count.setText(f"{peaks.size}")
        self.lineedit_diff.setText(f"{peaks.size - x * y}")

        self.updateImage()

    def cleanupPage(self) -> None:
        self.setField("peaks", self.field("peaks")[self.field("element")])

    def updateImage(self) -> None:
        if self.image is not None:
            self.graphics.scene().removeItem(self.image)

        peaks = self.field("peaks")

        x = int(self.lineedit_shape_x.text() or 0)
        y = int(self.lineedit_shape_y.text() or 0)

        if x == 0 or y == 0:
            return

        image = np.full((y, x), np.nan)
        image.flat = peaks[self.combo_element.currentText()][
            self.combo_integ.currentText()
        ]

        if self.check_raster.isChecked():
            image[::2, :] = image[::2, ::-1]

        table = get_table(self.graphics.options.colortable)
        self.image = ScaledImageItem.fromArray(
            image, QtCore.QRectF(0, 0, x, y), list(table)
        )
        self.graphics.scene().addItem(self.image)
        self.graphics.fitInView(QtCore.QRectF(0, 0, x, y))


class SpotConfigPage(QtWidgets.QWizardPage):
    dataChanged = QtCore.Signal()

    def __init__(self, config: SpotConfig, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setTitle("Elements and Config")

        self._datas: list[np.ndarray] = []
        self.label_elements = QtWidgets.QLabel()
        self.button_elements = QtWidgets.QPushButton("Edit Names")
        self.button_elements.pressed.connect(self.buttonNamesPressed)

        self.lineedit_spotsize_x = QtWidgets.QLineEdit()
        self.lineedit_spotsize_x.setText(str(config.spotsize))
        self.lineedit_spotsize_x.setValidator(DecimalValidatorNoZero(0, 1e9, 4))
        self.lineedit_spotsize_x.textChanged.connect(self.aspectChanged)
        self.lineedit_spotsize_x.textChanged.connect(self.completeChanged)

        self.lineedit_spotsize_y = QtWidgets.QLineEdit()
        self.lineedit_spotsize_y.setText(str(config.spotsize_y))
        self.lineedit_spotsize_y.setValidator(DecimalValidatorNoZero(0, 1e9, 4))
        self.lineedit_spotsize_y.textChanged.connect(self.aspectChanged)
        self.lineedit_spotsize_y.textChanged.connect(self.completeChanged)

        self.lineedit_aspect = QtWidgets.QLineEdit()
        self.lineedit_aspect.setEnabled(False)

        layout_elements = QtWidgets.QHBoxLayout()
        layout_elements.addWidget(QtWidgets.QLabel("Elements:"), 0, QtCore.Qt.AlignLeft)
        layout_elements.addWidget(self.label_elements, 1)
        layout_elements.addWidget(self.button_elements, 0, QtCore.Qt.AlignRight)

        config_box = QtWidgets.QGroupBox("Config")
        layout_config = QtWidgets.QFormLayout()
        layout_config.addRow("Spotsize X (μm):", self.lineedit_spotsize_x)
        layout_config.addRow("Spotsize Y (μm):", self.lineedit_spotsize_y)
        layout_config.addRow("Aspect:", self.lineedit_aspect)
        config_box.setLayout(layout_config)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(
            QtWidgets.QLabel("Edit imported elements and laser configuration."), 0
        )
        layout.addLayout(layout_elements)
        layout.addWidget(config_box)

        self.setLayout(layout)

        self.registerField("spotsize", self.lineedit_spotsize_x)
        self.registerField("spotsize_y", self.lineedit_spotsize_y)

    def getNames(self) -> list[str]:
        data = self.field("peaks")
        return data.dtype.names

    def initializePage(self) -> None:
        data = self.field("peaks")
        self.setElidedNames(data.dtype.names)

    def aspectChanged(self) -> None:
        try:
            aspect = float(self.field("spotsize")) / float(self.field("spotsize_y"))
            self.lineedit_aspect.setText(f"{aspect:.2f}")
        except (ValueError, ZeroDivisionError):
            self.lineedit_aspect.clear()

    def buttonNamesPressed(self) -> QtWidgets.QDialog:
        dlg = NameEditDialog(self.getNames(), allow_remove=True, parent=self)
        dlg.namesSelected.connect(self.updateNames)
        dlg.open()
        return dlg

    def isComplete(self) -> bool:
        if not self.lineedit_spotsize_x.hasAcceptableInput():
            return False
        if not self.lineedit_spotsize_y.hasAcceptableInput():
            return False
        return True

    def setElidedNames(self, names: list[str]) -> None:
        text = ", ".join(name for name in names)
        fm = QtGui.QFontMetrics(self.label_elements.font())
        text = fm.elidedText(text, QtCore.Qt.ElideRight, self.label_elements.width())
        self.label_elements.setText(text)

    def updateNames(self, rename: dict) -> None:
        peaks = self.field("peaks")
        remove = [name for name in peaks.dtype.names if name not in rename]
        peaks = rfn.drop_fields(peaks, remove, usemask=False)
        peaks = rfn.rename_fields(peaks, rename)

        self.setField("peaks", peaks)
        self.setElidedNames(peaks.dtype.names)
