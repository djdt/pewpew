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
from pewpew.widgets.dialogs import MultipleDirDialog

from typing import List, Tuple


class ImportFormatPage(QtWidgets.QWizardPage):
    def __init__(
        self, label: QtWidgets.QLabel = None, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)

        self.setTitle("Overview")
        if label is None:
            label = QtWidgets.QLabel()
        label.setWordWrap(True)

        mode_box = QtWidgets.QGroupBox("Data type", self)

        radio_numpy = QtWidgets.QRadioButton("&Numpy archives", self)
        radio_agilent = QtWidgets.QRadioButton("&Agilent batches", self)
        radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV exports", self)
        radio_numpy.setChecked(True)

        self.registerField("radio_numpy", radio_numpy)
        self.registerField("radio_agilent", radio_agilent)
        self.registerField("radio_thermo", radio_thermo)

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(radio_numpy)
        box_layout.addWidget(radio_agilent)
        box_layout.addWidget(radio_thermo)
        mode_box.setLayout(box_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(label)
        main_layout.addWidget(mode_box)
        self.setLayout(main_layout)


class LaserImportList(QtWidgets.QListWidget):
    def __init__(
        self, allowed_exts: List[str] = None, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.allowed_exts = allowed_exts
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setTextElideMode(QtCore.Qt.ElideLeft)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    path = url.toLocalFile()
                    name, ext = os.path.splitext(path)
                    if self.allowed_exts is None or ext in self.allowed_exts:
                        self.addItem(path)
        else:
            super().dropEvent(event)

    @QtCore.Property("QStringList")
    def paths(self) -> List[str]:
        return [self.item(i).text() for i in range(0, self.count())]


class ImportFilesPage(QtWidgets.QWizardPage):
    def __init__(
        self, min_files: int = 1, max_files: int = -1, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)

        self.min_files = min_files
        self.max_files = max_files

        self.setTitle("Files and Directories")
        # List and box

        self.list = LaserImportList()
        self.list.model().rowsInserted.connect(self.completeChanged)
        self.list.model().rowsRemoved.connect(self.completeChanged)

        dir_box = QtWidgets.QGroupBox("Layer Order", self)
        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(self.list)
        dir_box.setLayout(box_layout)

        # Buttons
        button_file = QtWidgets.QPushButton("Open")
        button_file.clicked.connect(self.buttonAdd)
        button_dir = QtWidgets.QPushButton("Open All...")
        button_dir.clicked.connect(self.buttonAddAll)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(button_file)
        button_layout.addWidget(button_dir)
        box_layout.addLayout(button_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(dir_box)
        self.setLayout(main_layout)

        self.registerField("paths", self.list, "paths")

    def initializePage(self) -> None:
        if self.field("radio_numpy"):
            ext = ".npz"
        elif self.field("radio_agilent"):
            ext = ".b"
        elif self.field("radio_thermo"):
            ext = ".csv"
        self.list.allowed_exts = [ext]

    def buttonAdd(self) -> None:
        if self.field("radio_numpy"):
            paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Select Files", "", "Numpy Archives(*.npz);;All Files(*)"
            )
        elif self.field("radio_agilent"):
            paths = MultipleDirDialog.getExistingDirectories(self, "Select Batches", "")
        elif self.field("radio_thermo"):
            paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Select Files", "", "CSV Documents(*.csv);;All Files(*)"
            )

        for path in paths:
            self.list.addItem(path)

    def buttonAddAll(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if len(path) == 0:
            return
        files = os.listdir(path)
        files.sort()

        ext = ".npz"
        if self.field("radio_agilent"):
            ext = ".b"
        elif self.field("radio_thermo"):
            ext = ".csv"

        for f in files:
            if f.lower().endswith(ext):
                self.list.addItem(os.path.join(path, f))

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if (
            event.key() == QtCore.Qt.Key_Delete
            or event.key() == QtCore.Qt.Key_Backspace
        ):
            for item in self.list.selectedItems():
                self.list.takeItem(self.list.row(item))
        super().keyPressEvent(event)

    def isComplete(self) -> bool:
        return self.list.count() >= self.min_files and (
            self.max_files < 0 or self.list.count() <= self.max_files
        )


class SpotImportWizard(QtWidgets.QWizard):
    laserImported = QtCore.Signal(Laser)

    def __init__(self, config: Config, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.config = config
        self.laser = None

        label = QtWidgets.QLabel(
            "Importer for data collected spotwise. To start, select the data type."
        )

        self.addPage(ImportFormatPage(label))
        self.addPage(ImportFilesPage(max_files=-1))
        self.addPage(SpotImportPeaksPage())


class SpotImportCanvas(BasicCanvas):
    def __init__(
        self,
        figsize: Tuple[float, float] = (5.0, 5.0),
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(figsize, parent)
        self.redrawFigure()

    def redrawFigure(self) -> None:
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


class SpotImportPeaksPage(QtWidgets.QWizardPage):
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
        self.paths = self.field("paths")
        if len(self.paths) == 1:
            self.check_single_spot.setChecked(True)

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


class SRRImportWizard(QtWidgets.QWizard):
    laserImported = QtCore.Signal(SRRLaser)

    def __init__(self, config: Config, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.config = SRRConfig(config.spotsize, config.speed, config.scantime)

        self.laser = None

        label = QtWidgets.QLabel(
            "This wizard will import SRR-LA-ICP-MS data. To begin, select "
            "the type of data to import. You may then import, reorder and "
            "configure the imported data."
        )

        self.addPage(ImportFormatPage(label))
        self.addPage(ImportFilesPage(min_files=2))
        self.addPage(SRRConfigPage(self.config))

        self.setWindowTitle("Kriss Kross Import Wizard")

        self.resize(540, 480)

    def accept(self) -> None:
        self.config.spotsize = float(self.field("spotsize"))
        self.config.speed = float(self.field("speed"))
        self.config.scantime = float(self.field("scantime"))
        self.config.warmup = float(self.field("warmup"))

        subpixel_width = self.field("subpixel_width")
        self.config.set_equal_subpixel_offsets(subpixel_width)

        paths = self.field("paths")
        layers = []

        if self.field("radio_numpy"):
            for path in paths:
                lds = io.npz.load(path)
                if len(lds) > 1:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Import Error",
                        f'Archive "{os.path.basename(path)}" '
                        "contains more than one image.",
                    )
                    return
                layers.append(lds[0].get())
        elif self.field("radio_agilent"):
            for path in paths:
                layers.append(io.agilent.load(path))
        elif self.field("radio_thermo"):
            for path in paths:
                layers.append(io.thermo.load(path))

        self.laserImported.emit(
            SRRLaser(
                layers,
                config=self.config,
                name=os.path.splitext(os.path.basename(paths[0]))[0],
                path=paths[0],
            )
        )
        super().accept()


class SRRConfigPage(QtWidgets.QWizardPage):
    def __init__(self, config: SRRConfig, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setText(str(config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidatorNoZero(0, 1e3, 4))
        self.lineedit_spotsize.textEdited.connect(self.completeChanged)

        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setText(str(config.speed))
        self.lineedit_speed.setValidator(DecimalValidatorNoZero(0, 1e3, 4))
        self.lineedit_speed.textEdited.connect(self.completeChanged)

        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setText(str(config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidatorNoZero(0, 1e3, 4))
        self.lineedit_scantime.textEdited.connect(self.completeChanged)

        # Krisskross params
        self.lineedit_warmup = QtWidgets.QLineEdit()
        self.lineedit_warmup.setText(str(config.warmup))
        self.lineedit_warmup.setValidator(DecimalValidator(0, 1e2, 2))
        self.lineedit_warmup.textEdited.connect(self.completeChanged)

        self.spinbox_offsets = QtWidgets.QSpinBox()
        self.spinbox_offsets.setRange(2, 10)
        self.spinbox_offsets.setValue(config._subpixel_size)
        self.spinbox_offsets.setToolTip(
            "The number of subpixels per pixel in each dimension."
        )

        # Form layout for line edits
        config_layout = QtWidgets.QFormLayout()
        config_layout.addRow("Spotsize (μm):", self.lineedit_spotsize)
        config_layout.addRow("Speed (μm):", self.lineedit_speed)
        config_layout.addRow("Scantime (s):", self.lineedit_scantime)

        config_gbox = QtWidgets.QGroupBox("Laser Configuration", self)
        config_gbox.setLayout(config_layout)

        params_layout = QtWidgets.QFormLayout()
        params_layout.addRow("Warmup (s):", self.lineedit_warmup)
        params_layout.addRow("Subpixel width:", self.spinbox_offsets)

        params_gbox = QtWidgets.QGroupBox("SRRLaser Parameters", self)
        params_gbox.setLayout(params_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(config_gbox)
        layout.addWidget(params_gbox)
        self.setLayout(layout)

        self.registerField("spotsize", self.lineedit_spotsize)
        self.registerField("speed", self.lineedit_speed)
        self.registerField("scantime", self.lineedit_scantime)
        self.registerField("warmup", self.lineedit_warmup)
        self.registerField("subpixel_width", self.spinbox_offsets)

    def isComplete(self) -> bool:
        return all(
            [
                self.lineedit_spotsize.hasAcceptableInput(),
                self.lineedit_speed.hasAcceptableInput(),
                self.lineedit_scantime.hasAcceptableInput(),
                self.lineedit_warmup.hasAcceptableInput(),
            ]
        )
