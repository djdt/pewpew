import numpy as np
import numpy.lib.recfunctions as rfn
import os

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib import io
from pewlib.config import Config
from pewlib.srr import SRRLaser, SRRConfig

from pewpew.validators import DecimalValidator

from pewpew.widgets.wizards.import_ import ConfigPage, FormatPage
from pewpew.widgets.wizards.options import PathAndOptionsPage

from typing import List, Tuple


class SRRPathAndOptionsPage(PathAndOptionsPage):
    def __init__(
        self,
        paths: List[str],
        format: str,
        nextid: int,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(
            paths, format, multiplepaths=True, nextid=nextid, parent=parent
        )

    def isComplete(self) -> bool:
        if not super().isComplete():
            return False
        return len(self.path.paths) >= 2


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
        self.setPage(
            self.page_agilent,
            SRRPathAndOptionsPage(
                paths, "agilent", nextid=self.page_config, parent=self
            ),
        )
        self.setPage(
            self.page_numpy,
            SRRPathAndOptionsPage(paths, "numpy", nextid=self.page_config, parent=self),
        )
        self.setPage(
            self.page_text,
            SRRPathAndOptionsPage(paths, "text", nextid=self.page_config, parent=self),
        )
        self.setPage(
            self.page_thermo,
            SRRPathAndOptionsPage(
                paths, "thermo", nextid=self.page_config, parent=self
            ),
        )

        self.setPage(self.page_config, SRRConfigPage(_config, parent=self))

    def accept(self) -> None:
        calibration = None
        if self.field("agilent"):
            path = self.field("agilent.paths")[0]
        elif self.field("numpy"):
            path = self.field("numpy.paths")[0]
            if self.field("numpy.useCalibration"):
                # Hack
                calibration = io.npz.load(path).calibration
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
        lasers = [io.npz.load(path) for path in paths]
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
