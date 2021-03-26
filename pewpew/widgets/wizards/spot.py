import numpy as np
import numpy.lib.recfunctions as rfn
import logging

from pathlib import Path

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib.config import Config
from pewlib.laser import Laser
from pewlib.process import peakfinding

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

    def generateThresholds(self) -> dict:
        raise NotImplementedError

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
        self.lineedit_maximum = QtWidgets.QLineEdit("1.0")
        self.lineedit_maximum.setValidator(DecimalValidator(-1e99, 1e99, 4))
        self.lineedit_maximum.textChanged.connect(self.optionsChanged)

        self.layout().addRow("Minimum value:", self.lineedit_minimum)
        self.layout().addRow("Maximum value:", self.lineedit_maximum)

    def args(self) -> dict:
        return {
            "minimum": float(self.lineedit_minimum.text()),
            "maximum": float(self.lineedit_maximum.text()),
        }

    def isComplete(self) -> bool:
        if (
            not self.lineedit_minimum.hasAcceptableInput()
            or not self.lineedit_maximum.hasAcceptableInput()
        ):
            return False
        return float(self.lineedit_minimum.text()) < float(self.lineedit_maximum.text())


class CWTPeakOptions(SpotPeakOptions):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("CWT Options", parent)
        self.lineedit_minwidth = QtWidgets.QLineEdit("10.0")
        self.lineedit_minwidth.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minwidth.textChanged.connect(self.optionsChanged)
        self.lineedit_maxwidth = QtWidgets.QLineEdit("30.0")
        self.lineedit_maxwidth.setValidator(DecimalValidator(1, 1e9, 2))
        self.lineedit_maxwidth.textChanged.connect(self.optionsChanged)

        self.lineedit_minsnr = QtWidgets.QLineEdit("9.0")
        self.lineedit_minsnr.setValidator(DecimalValidator(0, 100, 1))
        self.lineedit_minsnr.textChanged.connect(self.optionsChanged)
        self.lineedit_width_factor = QtWidgets.QLineEdit("2.5")
        self.lineedit_width_factor.setValidator(DecimalValidator(0, 10, 1))
        self.lineedit_width_factor.textChanged.connect(self.optionsChanged)

        self.combo_base_method = QtWidgets.QComboBox()
        self.combo_base_method.addItems(
            ["baseline", "edge", "prominence", "minima", "zero"]
        )
        self.combo_base_method.currentTextChanged.connect(self.optionsChanged)
        self.combo_height_method = QtWidgets.QComboBox()
        self.combo_height_method.addItems(["cwt", "maxima"])
        self.combo_height_method.currentTextChanged.connect(self.optionsChanged)

        self.layout().addRow("Minimum width:", self.lineedit_minwidth)
        self.layout().addRow("Maximum width:", self.lineedit_maxwidth)
        self.layout().addRow("Width factor:", self.lineedit_width_factor)
        self.layout().addRow("Minimum ridge SNR:", self.lineedit_minsnr)
        self.layout().addRow("Peak base:", self.combo_base_method)
        self.layout().addRow("Peak height:", self.combo_height_method)

    def args(self) -> dict:
        return {
            "width": (
                float(self.lineedit_minwidth.text()),
                float(self.lineedit_maxwidth.text()),
            ),
            "snr": float(self.lineedit_minsnr.text()),
            "width_factor": float(self.lineedit_width_factor.text()),
            "base_method": self.combo_base_method.currentText(),
            "height_method": self.combo_height_method.currentText(),
        }

    def isComplete(self) -> bool:
        if not all(
            x.hasAcceptableInput()
            for x in [
                self.lineedit_minwidth,
                self.lineedit_maxwidth,
                self.lineedit_minarea,
                self.lineedit_minheight,
                self.lineedit_minsnr,
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

        self.combo_filter_method = QtWidgets.QComboBox()
        self.combo_filter_method.addItems(["Guassian Mean", "Guassian Median"])
        self.combo_filter_method.currentTextChanged.connect(self.optionsChanged)

        self.lineedit_sigma = QtWidgets.QLineEdit("3.0")
        self.lineedit_sigma.setValidator(DecimalValidator(0.0, 100.0, 2))
        self.lineedit_sigma.textChanged.connect(self.optionsChanged)

        self.layout().addRow("Window size:", self.lineedit_window_size)
        self.layout().addRow("Window filter:", self.combo_filter_method)
        self.layout().addRow("Sigma:", self.lineedit_sigma)

    def args(self) -> dict:
        return {
            "size": int(self.lineedit_window_size.text()),
            "sigma": float(self.lineedit_sigma.text()),
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

        self.check_single_spot = QtWidgets.QCheckBox("One spot per line.")
        self.check_single_spot.clicked.connect(self.completeChanged)

        self.combo_peak_method = QtWidgets.QComboBox()
        self.combo_peak_method.addItems(list(self.options.keys()))

        self.combo_isotope = QtWidgets.QComboBox()

        self.stack = QtWidgets.QStackedWidget()
        for option in self.options.values():
            self.stack.addWidget(option)
            option.optionsChanged.connect(self.completeChanged)
            option.optionsChanged.connect(self.drawThresholds)

        self.combo_peak_method.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.combo_peak_method.currentIndexChanged.connect(self.drawThresholds)

        self.chart = SignalChart(parent=self)

        self.lineedit_minarea = QtWidgets.QLineEdit("0.0")
        self.lineedit_minarea.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minheight = QtWidgets.QLineEdit("0.0")
        self.lineedit_minheight.setValidator(DecimalValidator(0, 1e9, 2))

        # self.spinbox_line = QtWidgets.QSpinBox()
        # self.spinbox_line.setPrefix("line:")
        # self.spinbox_line.setValue(1)
        # self.spinbox_line.valueChanged.connect(self.updateCanvas)

        layout_form_controls = QtWidgets.QFormLayout()
        layout_form_controls.addRow("Minimum area:", self.lineedit_minarea)
        layout_form_controls.addRow("Minimum height:", self.lineedit_minheight)

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
        layout.addLayout(layout_chart)
        self.setLayout(layout)

        self.registerField("laserdatas", self, "data_prop")

    def completeChanged(self) -> None:
        pass

    def getData(self) -> List[np.ndarray]:
        return self._datas

    def setData(self, datas: List[np.ndarray]) -> None:
        self._datas = datas
        self.dataChanged.emit()

    def initializePage(self) -> None:
        datas = self.field("laserdatas")
        self.combo_isotope.clear()
        self.combo_isotope.addItems(datas[0].dtype.names)

        for signal in self.chart.signals.values():
            self.chart.removeSeries(signal)
        self.chart.signals.clear()

        data = datas[0][self.combo_isotope.currentText()].ravel()
        self.options["Constant"].lineedit_minimum.setText(f"{data.min():.2f}")
        self.options["Constant"].lineedit_maximum.setText(f"{data.max():.2f}")

        self.drawThresholds()

        self.chart.addSignal("signal", data)
        self.chart.yaxis.setRange(0, np.amax(data))
        self.chart.xaxis.setRange(0, data.size)
        self.chart.yaxis.applyNiceNumbers()

    def drawThresholds(self) -> None:
        method = self.combo_peak_method.currentText()
        args = self.options[method].args()
        data = self.field("laserdatas")[0][self.combo_isotope.currentText()].ravel()

        if method == "Constant":
            if "min" not in self.chart.signals:
                self.chart.addSignal(
                    "min", np.full(data.size, args["minimum"]), QtGui.QColor(0, 0, 255)
                )
            else:
                self.chart.setSignal("min", np.full(data.size, args["minimum"]))
            if "max" not in self.chart.signals:
                self.chart.addSignal(
                    "max", np.full(data.size, args["maximum"]), QtGui.QColor(255, 0, 0)
                )
            else:
                self.chart.setSignal("max", np.full(data.size, args["maximum"]))
        elif method == "CWT":
            pass
        elif method == "Moving window":
            pass

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
