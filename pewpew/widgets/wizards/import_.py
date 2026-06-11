import logging
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
from pewlib import io
from pewlib.config import Config, SpotConfig
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.validators import DecimalValidatorNoZero
from pewpew.widgets.dialogs import NameEditDialog
from pewpew.widgets.wizards.options import PathAndOptionsPage

logger = logging.getLogger(__name__)


class ImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_agilent = 1
    page_csv = 2
    page_perkinelmer = 3
    page_text = 4
    page_thermo = 5
    page_nu = 6
    page_config = 7

    laserImported = QtCore.Signal(Path, Laser)

    def __init__(
        self,
        path: str | Path = "",
        config: Config | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        if self.wizardStyle() != QtWidgets.QWizard.WizardStyle.MacStyle:
            self.setWizardStyle(QtWidgets.QWizard.WizardStyle.ModernStyle)
        self.setWindowTitle("Import Wizard")
        self.setMinimumSize(860, 680)

        if isinstance(path, str):  # pragma: no cover
            path = Path(path)

        config = config or SpotConfig()

        overview = (
            "This wizard will guide you through importing LA-ICP-MS data "
            "and provides a higher level to control than the standard import. "
            "To begin select the format of the file being imported."
        )

        format_page = FormatPage(
            overview,
            page_id_dict={
                "agilent": self.page_agilent,
                "csv": self.page_csv,
                "nu": self.page_nu,
                "numpy": -1,
                "perkinelmer": self.page_perkinelmer,
                "text": self.page_text,
                "thermo": self.page_thermo,
            },
            parent=self,
        )
        format_page.radio_numpy.setEnabled(False)
        format_page.radio_numpy.setVisible(False)

        self.setPage(self.page_format, format_page)
        self.setPage(
            self.page_agilent,
            PathAndOptionsPage([path], "agilent", nextid=self.page_config, parent=self),
        )
        self.setPage(
            self.page_csv,
            PathAndOptionsPage([path], "csv", nextid=self.page_config, parent=self),
        )
        self.setPage(
            self.page_perkinelmer,
            PathAndOptionsPage(
                [path], "perkinelmer", nextid=self.page_config, parent=self
            ),
        )
        self.setPage(
            self.page_text,
            PathAndOptionsPage([path], "text", nextid=self.page_config, parent=self),
        )
        self.setPage(
            self.page_thermo,
            PathAndOptionsPage([path], "thermo", nextid=self.page_config, parent=self),
        )
        self.setPage(
            self.page_nu,
            PathAndOptionsPage([path], "nu", nextid=self.page_config, parent=self),
        )

        self.setPage(self.page_config, ConfigPage(config, parent=self))

    def accept(self) -> None:
        if self.field("agilent"):
            paths = [Path(p) for p in self.field("agilent.paths")]
        elif self.field("csv"):
            paths = [Path(p) for p in self.field("csv.paths")]
        elif self.field("perkinelmer"):
            paths = [Path(p) for p in self.field("perkinelmer.paths")]
        elif self.field("text"):
            paths = [Path(p) for p in self.field("text.paths")]
        elif self.field("thermo"):
            paths = [Path(p) for p in self.field("thermo.paths")]
        elif self.field("nu"):
            paths = [Path(p) for p in self.field("nu.paths")]
        else:  # pragma: no cover
            raise ValueError("Invalid filetype selection.")

        datas = self.field("laserdata")
        infos = self.field("laserinfo")

        for path, data, info in zip(paths, datas, infos):
            config = SpotConfig(
                spotsize=float(self.field("spotsize")),
                spotsize_y=float(self.field("spotsize_y")),
            )
            self.laserImported.emit(path, Laser(data, config=config, info=info))
        super().accept()


class FormatPage(QtWidgets.QWizardPage):
    dataChanged = QtCore.Signal()
    paramsChanged = QtCore.Signal()
    infoChanged = QtCore.Signal()

    def __init__(
        self,
        text: str,
        page_id_dict: dict[str, int] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setTitle("Import Format")

        self.page_id_dict = page_id_dict

        self._laser_datas: list[np.ndarray] = []
        self._laser_params: list[dict] = []
        self._laser_infos: list[dict] = []

        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)

        self.radio_agilent = QtWidgets.QRadioButton("&Agilent Batches")
        self.radio_agilent.setChecked(True)
        self.radio_csv = QtWidgets.QRadioButton("&CSV Lines")
        self.radio_nu = QtWidgets.QRadioButton("N&u Vitesse Images")
        self.radio_numpy = QtWidgets.QRadioButton("&Numpy Archives")
        self.radio_perkinelmer = QtWidgets.QRadioButton("&Perkin-Elmer 'XL'")
        self.radio_text = QtWidgets.QRadioButton("Text and &CSV Images")
        self.radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV")

        format_box = QtWidgets.QGroupBox("File Format")
        layout_format = QtWidgets.QVBoxLayout()
        layout_format.addWidget(self.radio_agilent)
        layout_format.addWidget(self.radio_csv)
        layout_format.addWidget(self.radio_nu)
        layout_format.addWidget(self.radio_numpy)
        layout_format.addWidget(self.radio_perkinelmer)
        layout_format.addWidget(self.radio_text)
        layout_format.addWidget(self.radio_thermo)
        format_box.setLayout(layout_format)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(format_box)
        self.setLayout(layout)

        self.registerField("agilent", self.radio_agilent)
        self.registerField("csv", self.radio_csv)
        self.registerField("perkinelmer", self.radio_perkinelmer)
        self.registerField("nu", self.radio_nu)
        self.registerField("numpy", self.radio_numpy)
        self.registerField("text", self.radio_text)
        self.registerField("thermo", self.radio_thermo)

        self.registerField("laserdata", self, "data_prop")
        self.registerField("laserparam", self, "param_prop")
        self.registerField("laserinfo", self, "info_prop")

    def guessFormat(self, path: Path):
        if path.is_dir():
            if path.suffix == ".b":
                self.radio_agilent.setChecked(True)
            elif io.nu.is_nu_image_directory(path):
                self.radio_nu.setChecked(True)

    def initializePage(self) -> None:
        if self.page_id_dict is not None:
            self.radio_agilent.setVisible("agilent" in self.page_id_dict)
            self.radio_csv.setVisible("csv" in self.page_id_dict)
            self.radio_nu.setVisible("nu" in self.page_id_dict)
            self.radio_numpy.setVisible("numpy" in self.page_id_dict)
            self.radio_perkinelmer.setVisible("perkinelmer" in self.page_id_dict)
            self.radio_text.setVisible("text" in self.page_id_dict)
            self.radio_thermo.setVisible("thermo" in self.page_id_dict)

    def nextId(self) -> int:
        if self.page_id_dict is None:  # pragma: no cover
            return super().nextId()

        for field, page_id in self.page_id_dict.items():
            if self.field(field):
                return page_id

        return 0  # pragma: no cover

    def getData(self) -> list[np.ndarray]:
        return self._laser_datas

    def setData(self, datas: list[np.ndarray]) -> None:
        self._laser_datas = datas
        self.dataChanged.emit()

    def getParams(self) -> list[dict]:
        return self._laser_params

    def setParams(self, params: list[dict]) -> None:
        self._laser_params = params
        self.paramsChanged.emit()

    def getInfo(self) -> list[dict]:
        return self._laser_infos

    def setInfo(self, infos: list[dict]) -> None:
        self._laser_infos = infos
        self.infoChanged.emit()

    data_prop = QtCore.Property("QVariant", getData, setData, notify=dataChanged)  # type: ignore
    param_prop = QtCore.Property("QVariant", getParams, setParams, notify=paramsChanged)  # type: ignore
    info_prop = QtCore.Property("QVariant", getInfo, setInfo, notify=infoChanged)  # type: ignore


class ConfigPage(QtWidgets.QWizardPage):
    def __init__(self, config: Config, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setTitle("Elements and Config")

        self._datas: list[np.ndarray] = []
        self._infos: list[dict] = []

        if isinstance(config, SpotConfig):
            spotsize_y = config.spotsize_y
        else:
            spotsize_y = config.scantime * config.speed

        self.label_elements = QtWidgets.QLabel()
        self.button_elements = QtWidgets.QPushButton("Edit Names")
        self.button_elements.pressed.connect(self.buttonNamesPressed)

        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setText(str(config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidatorNoZero(0, 1e9, 4))
        self.lineedit_spotsize.setToolTip("Spot spacing in x direction in μm.")
        self.lineedit_spotsize.textChanged.connect(self.aspectChanged)
        self.lineedit_spotsize.textChanged.connect(self.completeChanged)

        self.lineedit_spotsize_y = QtWidgets.QLineEdit()
        self.lineedit_spotsize_y.setText(str(spotsize_y))
        self.lineedit_spotsize_y.setValidator(DecimalValidatorNoZero(0, 1e9, 4))
        self.lineedit_spotsize_y.setToolTip("Spot spacing in y direction in μm.")
        self.lineedit_spotsize_y.textChanged.connect(self.aspectChanged)
        self.lineedit_spotsize_y.textChanged.connect(self.completeChanged)

        self.lineedit_aspect = QtWidgets.QLineEdit()
        self.lineedit_aspect.setText(f"{config.spotsize / spotsize_y:.2f}")
        self.lineedit_aspect.setEnabled(False)

        layout_elements = QtWidgets.QHBoxLayout()
        layout_elements.addWidget(
            QtWidgets.QLabel("Elements:"), 0, QtCore.Qt.AlignmentFlag.AlignLeft
        )
        layout_elements.addWidget(self.label_elements, 1)
        layout_elements.addWidget(
            self.button_elements, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        config_box = QtWidgets.QGroupBox("Config")
        layout_config = QtWidgets.QFormLayout()
        layout_config.addRow("Spotsize X (μm):", self.lineedit_spotsize)
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

        self.registerField("spotsize", self.lineedit_spotsize)
        self.registerField("spotsize_y", self.lineedit_spotsize_y)

    def initializePage(self) -> None:
        params = self.field("laserparam")[0]
        data = self.field("laserdata")[0]

        self.setElidedNames(data.dtype.names)

        if "spotsize" in params:
            self.setField("spotsize", f"{params['spotsize']:.6g}")
        if "spotsize_y" in params:
            self.setField("spotsize_y", f"{params['spotsize_y']:.6g}")
        elif "speed" in params and "scantime" in params:
            self.setField("spotsize_y", f"{params['scantime'] * params['speed']:.6g}")

    def getNames(self) -> list[str]:
        data = self.field("laserdata")[0]
        return data.dtype.names if data is not None else []

    def aspectChanged(self) -> None:
        try:
            aspect = float(self.field("spotsize_y")) / float(self.field("spotsize"))
            self.lineedit_aspect.setText(f"{aspect:.2f}")
        except (ValueError, ZeroDivisionError):
            self.lineedit_aspect.clear()

    def buttonNamesPressed(self) -> QtWidgets.QDialog:
        dlg = NameEditDialog(self.getNames(), allow_remove=True, parent=self)
        dlg.namesSelected.connect(self.updateNames)
        dlg.open()
        return dlg

    def isComplete(self) -> bool:
        if not self.lineedit_spotsize.hasAcceptableInput():
            return False
        if not self.lineedit_spotsize_y.hasAcceptableInput():
            return False
        return True

    def setElidedNames(self, names: list[str]) -> None:
        text = ", ".join(name for name in names)
        fm = QtGui.QFontMetrics(self.label_elements.font())
        text = fm.elidedText(
            text, QtCore.Qt.TextElideMode.ElideRight, self.label_elements.width()
        )
        self.label_elements.setText(text)

    def updateNames(self, rename: dict) -> None:
        datas = self.field("laserdata")
        for i in range(len(datas)):
            remove = [name for name in datas[i].dtype.names if name not in rename]
            datas[i] = rfn.drop_fields(datas[i], remove, usemask=False)
            datas[i] = rfn.rename_fields(datas[i], rename)

        self.setField("laserdata", datas)
        self.setElidedNames(datas[0].dtype.names)
