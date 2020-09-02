import numpy as np
import numpy.lib.recfunctions as rfn
import os

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.config import Config
from pew.srr import SRRLaser, SRRConfig

from pewpew.actions import qAction, qToolButton
from pewpew.validators import DecimalValidator

from pewpew.widgets.wizards.import_ import FormatPage, ConfigPage
from pewpew.widgets.wizards.importoptions import (
    _ImportOptions,
    AgilentOptions,
    NumpyOptions,
    TextOptions,
    ThermoOptions,
)

from typing import List, Tuple


class SRRImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_agilent = 1
    page_numpy = 2
    page_text = 3
    page_thermo = 4
    page_config = 5

    laserImported = QtCore.Signal(SRRLaser)

    def __init__(
        self,
        paths: List[str] = [],
        config: Config = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SRR Import Wizard")

        _config = SRRConfig()
        if config is not None:
            _config.spotsize = config.spotsize
            _config.speed = config.speed
            _config.scantime = config.scantime

        overview = (
            "The wizard will guide you through importing LA-ICP-MS data "
            "and provides a higher level to control than the standard import. "
            "To begin select the format of the file being imported."
        )

        format_page = FormatPage(
            overview,
            page_id_dict={
                "agilent": self.page_agilent,
                "numpy": self.page_numpy,
                "text": self.page_text,
                "thermo": self.page_thermo,
            },
            parent=self,
        )

        self.setPage(self.page_format, format_page)
        self.setPage(self.page_agilent, SRRAgilentPage(paths, parent=self))
        self.setPage(self.page_numpy, SRRNumpyPage(paths, parent=self))
        self.setPage(self.page_text, SRRTextPage(paths, parent=self))
        self.setPage(self.page_thermo, SRRThermoPage(paths, parent=self))

        self.setPage(self.page_config, SRRConfigPage(_config, parent=self))

    def accept(self) -> None:
        calibration = None
        if self.field("agilent"):
            path = self.field("agilent.paths")[0]
        elif self.field("numpy"):
            path = self.field("numpy.paths")[0]
            if self.field("numpy.useCalibration"):
                # Hack
                calibration = io.npz.load(path)[0].calibration
        elif self.field("text"):
            path = self.field("text.paths")[0]
        elif self.field("thermo"):
            path = self.field("thermo.paths")[0]
        else:
            raise ValueError("Invalid filetype selection.")

        data = self.field("laserdata")
        config = SRRConfig(
            spotsize=float(self.field("spotsize")),
            scantime=float(self.field("scantime")),
            speed=float(self.field("speed")),
            warmup=float(self.field("warmup")),
        )
        config.set_equal_subpixel_offsets(self.field("subpixelWidth"))
        base, ext = os.path.splitext(path)
        self.laserImported.emit(
            SRRLaser(
                data,
                calibration=calibration,
                config=config,
                path=path,
                name=os.path.basename(base),
            )
        )
        super().accept()


class _SRRPageOptions(QtWidgets.QWizardPage):
    def __init__(
        self,
        options: _ImportOptions,
        paths: List[str] = [],
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setTitle(options.filetype + " Import")

        self.paths = PathsSelectionWidget(paths, options.filetype, options.exts)
        self.paths.pathsChanged.connect(self.pathsChanged)

        self.options = options
        self.options.completeChanged.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.paths, 0)
        layout.addWidget(self.options, 1)
        self.setLayout(layout)

    def buttonPathPressed(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(
            self, "Select File", os.path.dirname(self.lineedit_path.text())
        )
        dlg.setNameFilters([self.nameFilter(), "All Files(*)"])
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.fileSelected.connect(self.pathSelected)
        dlg.open()
        return dlg

    def isComplete(self) -> bool:
        paths = self.paths.paths
        if len(paths) < 2:
            return False
        for path in paths:
            if not self.validPath(path):
                return False
        return self.options.isComplete()

    def nameFilter(self) -> str:
        return f"{self.options.filetype}({' '.join(['*' + ext for ext in self.options.exts])})"

    def nextId(self) -> int:
        return SRRImportWizard.page_config

    def pathsChanged(self) -> None:
        self.completeChanged.emit()

    def validPath(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isfile(path)


class SRRAgilentPage(_SRRPageOptions):
    def __init__(self, paths: List[str] = [], parent: QtWidgets.QWidget = None):
        super().__init__(AgilentOptions(), paths, parent)
        self.paths.uses_directories = True

        self.registerField("agilent.paths", self.paths, "paths")
        self.registerField(
            "agilent.method",
            self.options.combo_dfile_method,
            "currentText",
            "currentTextChanged",
        )
        self.registerField("agilent.acqNames", self.options.check_name_acq_xml)

    def initializePage(self) -> None:
        self.pathsChanged()

    def pathsChanged(self) -> None:
        paths = self.field("agilent.paths")
        path = paths[0] if len(paths) > 0 else ""
        if self.validPath(path):
            self.options.setEnabled(True)
            self.options.updateOptions(path)
        else:
            self.options.actual_datafiles = 0
            self.options.setEnabled(False)
        super().pathsChanged()

    def validPath(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isdir(path)


class SRRNumpyPage(_SRRPageOptions):
    def __init__(self, paths: List[str] = [], parent: QtWidgets.QWidget = None):
        super().__init__(NumpyOptions(), paths, parent)
        self.registerField("numpy.paths", self.paths, "paths")
        self.registerField("numpy.useCalibration", self.options.check_calibration)


class SRRTextPage(_SRRPageOptions):
    def __init__(self, paths: List[str] = [], parent: QtWidgets.QWidget = None):
        super().__init__(TextOptions(), paths, parent)
        self.registerField("text.paths", self.paths, "paths")
        self.registerField("text.name", self.options.lineedit_name)


class SRRThermoPage(_SRRPageOptions):
    def __init__(self, paths: List[str] = [], parent: QtWidgets.QWidget = None):
        super().__init__(ThermoOptions(), paths, parent)

        self.registerField("thermo.paths", self.paths, "paths")
        self.registerField("thermo.sampleColumns", self.options.radio_columns)
        self.registerField("thermo.sampleRows", self.options.radio_rows)
        self.registerField(
            "thermo.delimiter",
            self.options.combo_delimiter,
            "currentText",
            "currentTextChanged",
        )
        self.registerField(
            "thermo.decimal",
            self.options.combo_decimal,
            "currentText",
            "currentTextChanged",
        )
        self.registerField("thermo.useAnalog", self.options.check_use_analog)

    def initializePage(self) -> None:
        self.pathsChanged()

    def pathChanged(self, path: str) -> None:
        paths = self.field("thermo.paths")
        path = paths[0] if len(paths) > 0 else ""
        if self.validPath(path):
            self.options.setEnabled(True)
            self.options.updateOptions(path)
        else:
            self.options.setEnabled(False)
        super().pathChanged(path)


class SRRConfigPage(ConfigPage):
    dataChanged = QtCore.Signal()

    def __init__(self, config: SRRConfig, parent: QtWidgets.QWidget = None):
        super().__init__(config, parent)
        self._srrdata: List[np.ndarray] = []

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

        params_box = QtWidgets.QGroupBox("SRR Parameters", self)
        params_layout = QtWidgets.QFormLayout()
        params_layout.addRow("Warmup (s):", self.lineedit_warmup)
        params_layout.addRow("Subpixel width:", self.spinbox_offsets)
        params_box.setLayout(params_layout)

        self.layout().addWidget(params_box)

        self.registerField("warmup", self.lineedit_warmup)
        self.registerField("subpixelWidth", self.spinbox_offsets)

    def getData(self) -> np.ndarray:
        return self._srrdata

    def setData(self, data: np.ndarray) -> None:
        self._srrdata = data
        self.dataChanged.emit()

    def initializePage(self) -> None:
        if self.field("agilent"):
            data, params = self.readSRRAgilent(self.field("agilent.paths"))
        elif self.field("numpy"):
            data, params = self.readSRRNumpy(self.field("numpy.paths"))
        elif self.field("text"):
            data, params = self.readSRRText(self.field("text.paths"))
        elif self.field("thermo"):
            data, params = self.readSRRThermo(self.field("thermo.paths"))

        if "scantime" in params:
            self.setField("scantime", f"{params['scantime']:.4g}")
        if "speed" in params:
            self.setField("speed", f"{params['speed']:.2g}")
        if "spotsize" in params:
            self.setField("spotsize", f"{params['spotsize']:.2g}")

        self.setField("laserdata", data)
        self.setElidedNames(data[0].dtype.names)

    def configValid(self) -> bool:
        data = self.field("laserdata")
        if len(data) < 2:
            return False
        spotsize = float(self.field("spotsize"))
        speed = float(self.field("speed"))
        scantime = float(self.field("scantime"))
        mag = np.round(spotsize / (speed * scantime)).astype(int)
        warmup = np.round(float(self.field("warmup")) / scantime).astype(int)
        if mag == 0:
            return False

        shape = data[1].shape[0], data[0].shape[1]
        limit = data[0].shape[0], data[1].shape[1]
        if mag * shape[0] + warmup > limit[0]:
            return False
        if mag * shape[1] + warmup > limit[1]:
            return False
        return True

    def getNames(self) -> List[str]:
        data = self.field("laserdata")[0]
        return data.dtype.names if data is not None else []

    def isComplete(self) -> bool:
        if not super().isComplete():
            return False
        if not self.lineedit_warmup.hasAcceptableInput():
            return False
        return self.configValid()

    def readSRRAgilent(self, paths: List[str]) -> Tuple[List[np.ndarray], dict]:
        data, param = self.readAgilent(paths[0])
        datas = [data]
        for path in paths[1:]:
            data, _ = self.readAgilent(path)
            datas.append(data)

        return datas, param

    def readSRRNumpy(self, paths: List[str]) -> Tuple[List[np.ndarray], dict]:
        lasers = [io.npz.load(path)[0] for path in paths]
        param = dict(
            scantime=lasers[0].config.scantime,
            speed=lasers[0].config.speed,
            spotsize=lasers[0].config.spotsize,
        )
        return [laser.data for laser in lasers], param

    def readSRRText(self, paths: List[str]) -> Tuple[np.ndarray, dict]:
        data, param = self.readAgilent(paths[0])
        datas = [data]
        for path in paths[1:]:
            data, _ = self.readText(path)
            datas.append(data)

        return data, param

    def readSRRThermo(self, paths: List[str]) -> Tuple[np.ndarray, dict]:
        data, param = self.readAgilent(paths[0])
        datas = [data]
        for path in paths[1:]:
            data, _ = self.readText(path)
            datas.append(data)

        return data, param

    def setElidedNames(self, names: List[str]) -> None:
        text = ", ".join(name for name in names)
        fm = QtGui.QFontMetrics(self.label_isotopes.font())
        text = fm.elidedText(text, QtCore.Qt.ElideRight, self.label_isotopes.width())
        self.label_isotopes.setText(text)

    def updateNames(self, rename: dict) -> None:
        datas = self.field("laserdata")
        for data in datas:
            remove = [name for name in data.dtype.names if name not in rename]
            data = rfn.drop_fields(data, remove, usemask=False)
            data = rfn.rename_fields(data, rename)

        self.setField("laserdata", datas)
        self.setElidedNames(datas[0].dtype.names)

    data_prop = QtCore.Property("QVariant", getData, setData, notify=dataChanged)


class PathsSelectionWidget(QtWidgets.QWidget):
    pathsChanged = QtCore.Signal()

    def __init__(
        self,
        files: List[str],
        filetype: str,
        exts: List[str],
        parent: QtWidgets.QWidget = None,
        uses_directories: bool = False,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.setTextElideMode(QtCore.Qt.ElideLeft)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.list.model().rowsInserted.connect(self.pathsChanged)
        self.list.model().rowsRemoved.connect(self.pathsChanged)
        self.list.model().rowsMoved.connect(self.pathsChanged)

        self.action_remove = qAction(
            "list-remove", "Remove", "Remove the selected path.", self.removeSelected
        )

        self.button_path = QtWidgets.QPushButton("Add Files")
        self.button_path.pressed.connect(self.buttonPathsPressed)
        self.button_dir = QtWidgets.QPushButton("Add All Files...")
        self.button_dir.pressed.connect(self.buttonDirectoryPressed)
        self.button_remove = qToolButton(action=self.action_remove)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button_path, 0, QtCore.Qt.AlignLeft)
        button_layout.addWidget(self.button_dir, 0, QtCore.Qt.AlignLeft)
        button_layout.addWidget(self.button_remove, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    @QtCore.Property("QStringList")
    def paths(self) -> List[str]:
        return [self.list.item(i).text() for i in range(0, self.list.count())]

    def addPath(self, path: str) -> None:
        self.list.addItem(path)

    def addPaths(self, paths: List[str]) -> None:
        for path in paths:
            self.list.addItem(path)

    def addPathsFromDirectory(self, directory: str) -> None:
        files = os.listdir(directory)
        files.sort()
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in self.exts:
                self.list.addItem(os.path.join(directory, f))

    def buttonPathsPressed(self) -> QtWidgets.QFileDialog:
        item = self.list.currentItem()
        dirname = os.path.dirname(item.text()) if item is not None else ""
        dlg = QtWidgets.QFileDialog(self, "Select Files", dirname)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        if self.uses_directories:
            dlg.setFileMode(QtWidgets.QFileDialog.Directory)
            dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
            dlg.fileSelected.connect(self.addPath)
        else:
            dlg.setNameFilters([self.nameFilter(), "All Files(*)"])
            dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
            dlg.fileSelected.connect(self.addPaths)
        dlg.open()
        return dlg

    def buttonDirectoryPressed(self) -> QtWidgets.QFileDialog:
        item = self.list.currentItem()
        dirname = os.path.dirname(item.text()) if item is not None else ""
        dlg = QtWidgets.QFileDialog(self, "Select Directory", dirname)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.addPathsFromDirectory)
        dlg.open()
        return dlg

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
                    if ext.lower() in self.exts:
                        self.list.addItem(path)
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if (
            event.key() == QtCore.Qt.Key_Delete
            or event.key() == QtCore.Qt.Key_Backspace
        ):
            self.removeSelected()
        super().keyPressEvent(event)

    def nameFilter(self) -> str:
        return f"{self.filetype}({' '.join(['*' + ext for ext in self.exts])})"

    def removeSelected(self) -> None:
        items = self.list.selectedItems()
        for item in items:
            self.list.takeItem(self.list.row(item))
