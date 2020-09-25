import os
import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.lib import peakfinding
from pew.config import Config
from pew.laser import Laser
from pew.srr import SRRLaser, SRRConfig

from pewpew.validators import DecimalValidator, DecimalValidatorNoZero
from pewpew.widgets.canvases import BasicCanvas

from pewpew.widgets.wizards.import_ import FormatPage
from pewpew.widgets.wizards.options import PathAndOptionsPage

from typing import Dict, List, Tuple


class SpotImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_files = 1
    page_agilent = 2
    page_numpy = 3
    page_text = 4
    page_thermo = 5
    page_spot_format = 6
    page_config = 7

    laserImported = QtCore.Signal(Laser)

    def __init__(
        self,
        paths: List[str] = [],
        config: Config = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Import Wizard")

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
                # "numpy": self.page_numpy,
                "text": self.page_text,
                "thermo": self.page_thermo,
            },
            parent=self,
        )
        format_page.radio_numpy.setEnabled(False)

        self.setPage(self.page_format, format_page)
        self.setPage(
            self.page_agilent,
            PathAndOptionsPage(
                paths,
                "agilent",
                multiplepaths=True,
                nextid=self.page_spot_format,
                parent=self,
            ),
        )
        self.setPage(
            self.page_text,
            PathAndOptionsPage(
                paths,
                "text",
                multiplepaths=True,
                nextid=self.page_spot_format,
                parent=self,
            ),
        )
        self.setPage(
            self.page_thermo,
            PathAndOptionsPage(
                paths,
                "thermo",
                multiplepaths=True,
                nextid=self.page_spot_format,
                parent=self,
            ),
        )
        self.setPage(self.page_spot_format, SpotPeakFindingPage(parent=self))

    def accept(self) -> None:
        if self.field("agilent"):
            path = self.field("agilent.path")
        elif self.field("text"):
            path = self.field("text.path")
        elif self.field("thermo"):
            path = self.field("thermo.path")
        else:
            raise ValueError("Invalid filetype selection.")

        data = self.field("laserdata")
        config = Config(
            spotsize=float(self.field("spotsize")),
            scantime=float(self.field("scantime")),
            speed=float(self.field("speed")),
        )
        base, ext = os.path.splitext(path)
        self.laserImported.emit(
            Laser(data, config=config, path=path, name=os.path.basename(base))
        )
        super().accept()


# class SpotFormatPage(QtWidgets.QWizardPage):
#     def __init__(
#         self, parent: QtWidgets.QWidget = None,
#     ):
#         super().__init__(parent)
#         self.radio_single_line = QtWidgets.QRadioButton("Single acquistion.")
#         self.radio_multi_line = QtWidgets.QRadioButton("Multiple acquistions.")
#         self.radio_spot_per_line = QtWidgets.QRadioButton("One spot per acquistion.")

#         self.lineedit_shape_x = QtWidgets.QLineEdit("1")
#         self.lineedit_shape_x.setValidator(QtGui.QIntValidator(1, 9999))
#         self.lineedit_shape_x.textEdited.connect(self.completeChanged)
#         self.lineedit_shape_y = QtWidgets.QLineEdit("1")
#         self.lineedit_shape_y.setValidator(QtGui.QIntValidator(1, 9999))
#         self.lineedit_shape_y.textEdited.connect(self.completeChanged)

#         self.check_rastered = QtWidgets.QCheckBox("Data is rastered.")

#         spot_box = QtWidgets.QGroupBox("Spot Format")
#         layout_spot = QtWidgets.QVBoxLayout()
#         layout_spot.addWidget(self.radio_single_line)
#         layout_spot.addWidget(self.radio_multi_line)
#         layout_spot.addWidget(self.radio_spot_per_line)
#         spot_box.setLayout(layout_spot)

#         layout_shape = QtWidgets.QHBoxLayout()
#         layout_shape.addWidget(self.lineedit_shape_x)
#         layout_shape.addWidget(QtWidgets.QLabel("x"))
#         layout_shape.addWidget(self.lineedit_shape_y)

#         layout_options = QtWidgets.QFormLayout()
#         layout_options.addRow("Shape:", layout_shape)
#         layout_options.addRow(self.check_rastered)

#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(spot_box)
#         layout.addLayout(layout_options)
#         self.setLayout(layout)

#         self.registerField("singleLine", self.radio_single_line)
#         self.registerField("multiLine", self.radio_multi_line)
#         self.registerField("spotPerLine", self.radio_spot_per_line)

#         self.registerField("shapeX", self.lineedit_shape_x)
#         self.registerField("shapeY", self.lineedit_shape_y)
#         self.registerField("rastered", self.check_rastered)

#     def nextId(self) -> int:
#         if self.field("singleLine"):
#             return SpotImportWizard.page_single_line
#         elif self.field("multiLine"):
#             return SpotImportWizard.page_multi_line
#         elif self.field("spotPerLine"):
#             return SpotImportWizard.page_spot_per_line
#         else:
#             return super().nextId()


class SpotImportCanvas(BasicCanvas):
    def __init__(
        self,
        figsize: Tuple[float, float] = (5.0, 5.0),
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(figsize, parent)
        self.drawFigure()

    def drawFigure(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot()

    def drawPeaks(self, data: np.ndarray, peaks: np.ndarray) -> None:
        self.ax.clear()
        self.ax.plot(data, color="black")

        for peak in peaks:
            xs = np.arange(peak["left"], peak["right"])
            self.ax.fill_between(
                xs, data[xs], peak["base"], color="red", alpha=0.5, lw=0.5
            )

        self.ax.text(
            0.05,
            0.95,
            str(len(peaks)),
            ha="left",
            va="top",
            transform=self.ax.transAxes,
        )

        self.draw_idle()


class SpotPeakFindingPage(QtWidgets.QWizardPage):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.check_single_spot = QtWidgets.QCheckBox("One spot per line.")
        self.check_single_spot.clicked.connect(self.completeChanged)

        self.lineedit_shape_x = QtWidgets.QLineEdit("1")
        self.lineedit_shape_x.setValidator(QtGui.QIntValidator(1, 9999))
        self.lineedit_shape_x.textEdited.connect(self.completeChanged)
        self.lineedit_shape_y = QtWidgets.QLineEdit("1")
        self.lineedit_shape_y.setValidator(QtGui.QIntValidator(1, 9999))
        self.lineedit_shape_y.textEdited.connect(self.completeChanged)

        self.lineedit_minwidth = QtWidgets.QLineEdit("10.0")
        self.lineedit_minwidth.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minwidth.editingFinished.connect(self.updateCanvas)
        self.lineedit_maxwidth = QtWidgets.QLineEdit("30.0")
        self.lineedit_maxwidth.setValidator(DecimalValidator(1, 1e9, 2))
        self.lineedit_maxwidth.editingFinished.connect(self.updateCanvas)

        box_params = QtWidgets.QGroupBox("Peak Finding")

        self.lineedit_minarea = QtWidgets.QLineEdit("0.0")
        self.lineedit_minarea.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minarea.editingFinished.connect(self.updateCanvas)
        self.lineedit_minheight = QtWidgets.QLineEdit("0.0")
        self.lineedit_minheight.setValidator(DecimalValidator(0, 1e9, 2))
        self.lineedit_minheight.editingFinished.connect(self.updateCanvas)
        self.lineedit_minsnr = QtWidgets.QLineEdit("9.0")
        self.lineedit_minsnr.setValidator(DecimalValidator(0, 100, 1))
        self.lineedit_minsnr.editingFinished.connect(self.updateCanvas)
        self.lineedit_width_factor = QtWidgets.QLineEdit("2.5")
        self.lineedit_width_factor.setValidator(DecimalValidator(0, 10, 1))
        self.lineedit_width_factor.editingFinished.connect(self.updateCanvas)

        self.combo_base_method = QtWidgets.QComboBox()
        self.combo_base_method.addItems(
            ["baseline", "edge", "prominence", "minima", "zero"]
        )
        self.combo_base_method.activated.connect(self.updateCanvas)
        self.combo_height_method = QtWidgets.QComboBox()
        self.combo_height_method.addItems(["cwt", "maxima"])
        self.combo_height_method.activated.connect(self.updateCanvas)

        self.canvas = SpotImportCanvas(parent=self)

        self.spinbox_line = QtWidgets.QSpinBox()
        self.spinbox_line.setPrefix("line:")
        self.spinbox_line.setValue(1)
        self.spinbox_line.valueChanged.connect(self.updateCanvas)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.activated.connect(self.updateCanvas)

        layout_shape = QtWidgets.QHBoxLayout()
        layout_shape.addWidget(QtWidgets.QLabel("Shape:"))
        layout_shape.addWidget(self.lineedit_shape_x)
        layout_shape.addWidget(QtWidgets.QLabel("x"))
        layout_shape.addWidget(self.lineedit_shape_y)

        layout_width = QtWidgets.QHBoxLayout()
        layout_width.addWidget(QtWidgets.QLabel("Width:"))
        layout_width.addWidget(self.lineedit_minwidth)
        layout_width.addWidget(QtWidgets.QLabel("-"))
        layout_width.addWidget(self.lineedit_maxwidth)

        box_layout = QtWidgets.QFormLayout()
        box_layout.addRow("Minimum ridge SNR:", self.lineedit_minsnr)
        box_layout.addRow("Minimum area:", self.lineedit_minarea)
        box_layout.addRow("Minimum height:", self.lineedit_minheight)
        box_layout.addRow("Peak base:", self.combo_base_method)
        box_layout.addRow("Peak height:", self.combo_height_method)
        box_layout.addRow("Width factor:", self.lineedit_width_factor)
        box_params.setLayout(box_layout)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addWidget(self.check_single_spot)
        layout_controls.addLayout(layout_shape)
        layout_controls.addLayout(layout_width)
        layout_controls.addWidget(box_params)

        layout_combos = QtWidgets.QHBoxLayout()
        layout_combos.addWidget(self.spinbox_line, 0, QtCore.Qt.AlignLeft)
        layout_combos.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignRight)

        layout_canvas = QtWidgets.QVBoxLayout()
        layout_canvas.addWidget(self.canvas)
        layout_canvas.addLayout(layout_combos)

        layout_main = QtWidgets.QHBoxLayout()
        layout_main.addLayout(layout_controls)
        layout_main.addLayout(layout_canvas)
        self.setLayout(layout_main)

    def initializePage(self) -> None:
        # self.paths = self.field("paths")
        # if len(self.paths) == 1:
        #     self.check_single_spot.setChecked(True)

        self.updateData(0)

        self.spinbox_line.setRange(1, self.data.shape[0])
        self.combo_isotope.addItems(self.data.dtype.names)

        self.updateCanvas()

    def isComplete(self) -> bool:
        if self.check_single_spot.isChecked():
            return int(self.lineedit_shape_x.text()) * int(
                self.lineedit_shape_y.text()
            ) == len(self.paths)
        return True

    @QtCore.Property(dict)
    def peakParams(self) -> dict:
        return {
            "peak_base_method": self.combo_base_method.currentText(),
            "peak_height_method": self.combo_height_method.currentText(),
            "peak_width_factor": float(self.lineedit_width_factor.text()),
            "peak_min_area": float(self.lineedit_minarea.text()),
            "peak_min_height": float(self.lineedit_minheight.text()),
            "ridge_min_snr": float(self.lineedit_minsnr.text()),
        }

    def readAgilent(self, paths: List[str]) -> List[np.ndarray]:
        data = []
        for path in paths:
            agilent_method = self.field("agilent.method")
            if agilent_method == "Alphabetical Order":
                method = None
            elif agilent_method == "Acquistion Method":
                method = ["acq_method_xml"]
            elif agilent_method == "Batch Log CSV":
                method = ["batch_csv"]
            elif agilent_method == "Batch Log XML":
                method = ["batch_xml"]
            else:
                raise ValueError("Unknown data file collection method.")

            data.append(
                io.agilent.load(
                    path,
                    collection_methods=method,
                    use_acq_for_names=self.field("agilent.useAcqNames"),
                )
            )
        return data

    def readText(self, paths: List[str]) -> List[np.ndarray]:
        data = []
        for path in paths:
            data.append(io.csv.load(path, isotope=self.field("text.name")))
        return data

    def readThermo(self, paths: List[str]) -> List[np.ndarray]:
        data = []
        for path in paths:
            kwargs = dict(
                delimiter=self.field("thermo.delimiter"),
                comma_decimal=self.field("thermo.decimal") == ",",
            )
            use_analog = self.field("thermo.useAnalog")

            if self.field("thermo.sampleRows"):
                data.append(
                    io.thermo.icap_csv_rows_read_data(
                        path, use_analog=use_analog, **kwargs
                    )
                )
            else:
                data.append(
                    io.thermo.icap_csv_columns_read_data(
                        path, use_analog=use_analog, **kwargs
                    )
                )
        return data

    def updateCanvas(self) -> None:
        data = self.data[self.combo_isotope.currentText()][
            self.spinbox_line.value() - 1
        ]
        min_width = float(self.lineedit_minwidth.text())
        max_width = float(self.lineedit_maxwidth.text())
        peaks = peakfinding.find_peaks(
            data, int(min_width), int(max_width), **self.peakParams
        )
        self.canvas.drawPeaks(data, peaks)

    def updateData(self, idx: int = 0) -> None:
        path = self.paths[0]
        if self.field("radio_numpy"):
            self.data = io.npz.load(path)[0].get(calibrated=None, flat=True)
        elif self.field("radio_agilent"):
            self.data = io.agilent.load(path)
        elif self.field("radio_thermo"):
            self.data = io.thermo.load(path)
        else:
            raise ValueError("No radio selected!")


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    w = SpotImportWizard(["/home/tom/Downloads/spotty boy.b"])
    w.show()
    app.exec_()
