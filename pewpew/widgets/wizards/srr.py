from pathlib import Path
import time

from PySide6 import QtCore, QtWidgets

from pewlib import __version__ as pewlib_version
from pewlib import io
from pewlib.config import Config
from pewlib.srr import SRRLaser, SRRConfig

from pewpew import __version__ as pewpew_version
from pewpew.validators import DecimalValidator

from pewpew.widgets.wizards.import_ import ConfigPage, FormatPage
from pewpew.widgets.wizards.options import PathAndOptionsPage

from typing import List, Optional


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
        paths: List[Path] = [],
        config: Optional[Config] = None,
        parent: Optional[QtWidgets.QWidget] = None,
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
            path = Path(self.field("agilent.paths")[0])
        elif self.field("numpy"):
            path = Path(self.field("numpy.paths")[0])
            if self.field("numpy.useCalibration"):  # pragma: no cover
                # Hack
                calibration = io.npz.load(path).calibration
        elif self.field("text"):
            path = Path(self.field("text.paths")[0])
        elif self.field("thermo"):
            path = Path(self.field("thermo.paths")[0])
        else:  # pragma: no cover
            raise ValueError("Invalid filetype selection.")

        datas = self.field("laserdata")
        config = SRRConfig(
            spotsize=float(self.field("spotsize")),
            scantime=float(self.field("scantime")),
            speed=float(self.field("speed")),
            warmup=float(self.field("warmup")),
        )
        config.set_equal_subpixel_offsets(self.field("subpixelWidth"))

        info = self.field("laserinfo")[0]
        info.update(
            {
                "Name": path.stem,
                "File Path": str(path.resolve()),
                "Import Date": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())
                ),
                "Import Path": str(path.resolve()),
                "Import Version pewlib": pewlib_version,
                "Import Version pew2": pewpew_version,
            }
        )
        self.laserImported.emit(
            SRRLaser(
                datas,
                calibration=calibration,
                config=config,
                info={"Name": path.stem, "File Path": str(path.resolve())}
            )
        )
        super().accept()


class SRRConfigPage(ConfigPage):
    def __init__(self, config: SRRConfig, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(config, parent)

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

    def configValid(self) -> bool:
        datas = self.field("laserdata")
        if len(datas) < 2:
            return False
        spotsize = float(self.field("spotsize"))
        speed = float(self.field("speed"))
        scantime = float(self.field("scantime"))
        warmup = float(self.field("warmup"))
        config = SRRConfig(
            spotsize=spotsize, speed=speed, scantime=scantime, warmup=warmup
        )
        return config.valid_for_data(datas)

    def isComplete(self) -> bool:
        if not super().isComplete():
            return False
        if not self.lineedit_warmup.hasAcceptableInput():
            return False
        return self.configValid()


class SRRPathAndOptionsPage(PathAndOptionsPage):
    def __init__(
        self,
        paths: List[Path],
        format: str,
        nextid: int,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(
            paths, format, multiplepaths=True, nextid=nextid, parent=parent
        )

    def isComplete(self) -> bool:
        if not super().isComplete():  # pragma: no cover
            return False
        return len(self.path.paths) >= 2
