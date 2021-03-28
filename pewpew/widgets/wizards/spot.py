import numpy as np
import numpy.lib.recfunctions as rfn
import logging

from pathlib import Path

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

from pewlib.config import Config
from pewlib.laser import Laser
from pewlib.process import peakfinding
from pewlib.process.calc import view_as_blocks

from pewpew.charts.signal import SignalChart
from pewpew.validators import DecimalValidator, OddIntValidator

from pewpew.widgets.wizards.import_ import FormatPage
from pewpew.widgets.wizards.options import PathAndOptionsPage

from typing import Dict, List, Union


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

    laserImported = QtCore.Signal(Laser)

    def __init__(
        self,
        paths: Union[List[str], List[Path]] = [],
        config: Config = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spot Import Wizard")

        for i in enumerate(paths):
            if isinstance(paths[0], str):  # pragma: no cover
                paths[0] = Path(paths[0])

        config = config or Config()

        overview = (
            "The wizard will guide you through importing LA-ICP-MS data "
            "and provides a higher level to control than the standard import. "
            "To begin select the format of the file being imported."
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
                paths, "agilent", nextid=self.page_spot_peaks, parent=self
            ),
        )
        self.setPage(
            self.page_csv,
            PathAndOptionsPage(paths, "csv", nextid=self.page_spot_peaks, parent=self),
        )
        self.setPage(
            self.page_numpy,
            PathAndOptionsPage(
                paths, "numpy", nextid=self.page_spot_peaks, parent=self
            ),
        )
        self.setPage(
            self.page_perkinelmer,
            PathAndOptionsPage(
                paths, "perkinelmer", nextid=self.page_spot_peaks, parent=self
            ),
        )
        self.setPage(
            self.page_text,
            PathAndOptionsPage(paths, "text", nextid=self.page_spot_peaks, parent=self),
        )
        self.setPage(
            self.page_thermo,
            PathAndOptionsPage(
                paths, "thermo", nextid=self.page_spot_peaks, parent=self
            ),
        )

        self.setPage(self.page_spot_peaks, SpotPeaksPage(parent=self))

    def accept(self) -> None:
        # if self.field("agilent"):
        #     path = Path(self.field("agilent.path"))
        # elif self.field("csv"):
        #     path = Path(self.field("csv.path"))
        # elif self.field("perkinelmer"):
        #     path = Path(self.field("perkinelmer.path"))
        # elif self.field("text"):
        #     path = Path(self.field("text.path"))
        # elif self.field("thermo"):
        #     path = Path(self.field("thermo.path"))
        # else:  # pragma: no cover
        #     raise ValueError("Invalid filetype selection.")

        # data = self.field("laserdata")
        # config = Config(
        #     spotsize=float(self.field("spotsize")),
        #     scantime=float(self.field("scantime")),
        #     speed=float(self.field("speed")),
        # )
        # self.laserImported.emit(Laser(data, config=config, name=path.stem, path=path))
        super().accept()


class SpotPeakOptions(QtWidgets.QGroupBox):
    optionsChanged = QtCore.Signal()

    def __init__(self, name: str, parent: QtWidgets.QWidget = None):
        super().__init__(name, parent=parent)
        self.setLayout(QtWidgets.QFormLayout())

    # def fieldArgs(self) -> List[Tuple[str, QtWidgets.QWidget, str, str]]:
    #     return []

    def isComplete(self) -> bool:
        return True

    # def setEnabled(self, enabled: bool) -> None:
    #     pass

    # def updateForPath(self, path: Path) -> None:
    #     pass


class ConstantPeakOptions(SpotPeakOptions):
    def __init__(self, parent: QtWidgets.QWidget = None):
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
    def __init__(self, parent: QtWidgets.QWidget = None):
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
        self.lineedit_minsnr.setValidator(DecimalValidator(0, 100, 1))
        self.lineedit_minsnr.textChanged.connect(self.optionsChanged)
        self.lineedit_width_factor = QtWidgets.QLineEdit("2.5")
        self.lineedit_width_factor.setValidator(DecimalValidator(0, 10, 1))
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
    def __init__(self, parent: QtWidgets.QWidget = None):
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
    dataChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self._datas: List[np.ndarray] = []
        self.options = {
            "Constant": ConstantPeakOptions(self),
            "CWT": CWTPeakOptions(self),
            "Moving window": WindowedPeakOptions(self),
        }
        self.peaks: np.ndarray = None

        self.check_single_spot = QtWidgets.QCheckBox("One spot per line.")
        self.check_single_spot.clicked.connect(self.completeChanged)

        self.combo_peak_method = QtWidgets.QComboBox()
        self.combo_peak_method.addItems(list(self.options.keys()))

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.onIsotopeChanged)

        self.stack = QtWidgets.QStackedWidget()
        for option in self.options.values():
            self.stack.addWidget(option)
            option.optionsChanged.connect(self.completeChanged)
            option.optionsChanged.connect(self.updatePeaks)

        self.combo_peak_method.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.combo_peak_method.currentIndexChanged.connect(self.updatePeaks)

        self.chart = SignalChart(parent=self)

        self.combo_base_method = QtWidgets.QComboBox()
        self.combo_base_method.addItems(
            ["baseline", "edge", "prominence", "minima", "zero"]
        )
        # self.combo_base_method.currentTextChanged.connect(self.drawThresholds)
        self.combo_height_method = QtWidgets.QComboBox()
        self.combo_height_method.addItems(["center", "maxima"])
        self.combo_height_method.setCurrentText("maxima")
        # self.combo_height_method.currentTextChanged.connect(self.optionsChanged)

        self.lineedit_minarea = QtWidgets.QLineEdit("0.0")
        self.lineedit_minarea.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minheight = QtWidgets.QLineEdit("0.0")
        self.lineedit_minheight.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minwidth = QtWidgets.QLineEdit("0.0")
        self.lineedit_minwidth.setValidator(DecimalValidator(0, 1e9, 2))

        # self.spinbox_line = QtWidgets.QSpinBox()
        # self.spinbox_line.setPrefix("line:")
        # self.spinbox_line.setValue(1)
        # self.spinbox_line.valueChanged.connect(self.updateCanvas)

        layout_form_controls = QtWidgets.QFormLayout()
        layout_form_controls.addRow("Peak base:", self.combo_base_method)
        layout_form_controls.addRow("Peak height:", self.combo_height_method)
        layout_form_controls.addRow("Minimum area:", self.lineedit_minarea)
        layout_form_controls.addRow("Minimum height:", self.lineedit_minheight)
        layout_form_controls.addRow("Minimum width:", self.lineedit_minwidth)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addWidget(self.check_single_spot)
        layout_controls.addWidget(self.combo_peak_method)
        layout_controls.addWidget(self.stack)
        layout_controls.addLayout(layout_form_controls)

        layout_chart = QtWidgets.QVBoxLayout()
        layout_chart.addWidget(self.chart, 1)
        layout_chart.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_controls)
        layout.addLayout(layout_chart, 1)
        self.setLayout(layout)

        self.registerField("laserdatas", self, "data_prop")

    def completeChanged(self) -> None:
        pass

    def isComplete(self) -> None:
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

    def getData(self) -> List[np.ndarray]:
        return self._datas

    def setData(self, datas: List[np.ndarray]) -> None:
        self._datas = datas
        self.dataChanged.emit()

    def initializePage(self) -> None:
        datas = self.field("laserdatas")
        self.combo_isotope.clear()
        self.combo_isotope.addItems(datas[0].dtype.names)

        data = datas[0][self.combo_isotope.currentText()].ravel()
        self.options["Constant"].lineedit_minimum.setText(
            f"{np.percentile(data, 25):.2f}"
        )

        self.drawSignal(data)
        self.updatePeaks()

    def onIsotopeChanged(self) -> None:
        data = self.field("laserdatas")[0][self.combo_isotope.currentText()].ravel()
        self.drawSignal(data)
        self.updatePeaks()

    def drawSignal(self, data: np.ndarray) -> None:
        if "signal" in self.chart.series:
            self.chart.setSeries("signal", data)
        else:
            self.chart.addSeries("signal", data)
            self.chart.yaxis.setRange(0, np.amax(data))
            self.chart.xaxis.setRange(0, data.size)
            self.chart.yaxis.applyNiceNumbers()

    def clearPeaks(self) -> None:
        if "peaks" in self.chart.series:
            self.chart.series.pop("peaks").clear()

    def drawPeaks(self, peaks: np.ndarray) -> None:
        if "peaks" in self.chart.series:
            self.chart.setSeries("peaks", peaks["height"] + peaks["base"], peaks["top"])
        else:
            self.chart.addSeries(
                "peaks",
                peaks["height"] + peaks["base"],
                peaks["top"],
                series_type=QtCharts.QScatterSeries,
                color=QtGui.QColor(255, 0, 0),
            )

    def clearThresholds(self) -> None:
        for name in list(self.chart.series.keys()):
            if name not in ["signal", "peaks"]:
                self.chart.chart().removeSeries(self.chart.series.pop(name))

    def updatePeaks(self) -> None:
        if not self.isComplete():
            self.clearThresholds()
            self.clearPeaks()
            return

        method = self.combo_peak_method.currentText()
        args = self.options[method].args()
        data = self.field("laserdatas")[0][self.combo_isotope.currentText()].ravel()

        if method == "Constant":
            thresholds = {"minimum": np.full(data.size, args["minimum"])}
            diff = np.diff((data > thresholds["minimum"]).astype(np.int8), prepend=0)
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
                self.peaks = None
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

            if args["thresh"] == "Std":
                threshold = np.std(view, axis=1) * args["value"]
            elif args["thresh"] == "Constant":
                threshold = np.full(data.size, args["value"])

            thresholds = {"baseline": baseline, "threshold": baseline + threshold}

            diff = np.diff((data > (baseline + threshold)).astype(np.int8), prepend=0)
            lefts = np.flatnonzero(diff == 1)
            rights = np.flatnonzero(diff == -1)
            if rights.size > lefts.size:
                rights = rights[1:]
            elif lefts.size > rights.size:
                lefts = lefts[:-1]

        colors = iter(
            [QtGui.QColor(0, 0, 255), QtGui.QColor(255, 0, 0), QtGui.QColor(0, 255, 0)]
        )

        for name, value in thresholds.items():
            if name in self.chart.series:
                self.chart.setSeries(name, value)
            else:
                self.chart.addSeries(name, value, color=next(colors))

        if lefts.size == 0 or rights.size == 0:
            self.peaks = None
            self.clearPeaks()
            return

        self.peaks = peakfinding.peaks_from_edges(
            data,
            lefts,
            rights,
            base_method=self.combo_base_method.currentText(),
            height_method=self.combo_height_method.currentText(),
        )

        self.peaks = peakfinding.filter_peaks(
            self.peaks,
            min_area=float(self.lineedit_minarea.text()),
            min_height=float(self.lineedit_minheight.text()),
            min_width=float(self.lineedit_minwidth.text()),
        )

        self.drawPeaks(self.peaks)

    data_prop = QtCore.Property("QVariant", getData, setData, notify=dataChanged)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    spot = SpotImportWizard(
        [
            "/home/tom/MEGA/Uni/Experimental/LAICPMS/Standards/IDA agar/20200815_gel_ida_iso_brain_spot_63x54.b"
        ]
    )
    spot.open()
    spot.next()
    spot.next()
    app.exec_()
