import numpy as np
import numpy.lib.recfunctions as rfn
import logging

from pathlib import Path

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib.config import Config
from pewlib.laser import Laser

from pewpew.validators import DecimalValidatorNoZero
from pewpew.widgets.dialogs import NameEditDialog
from pewpew.widgets.wizards.options import PathAndOptionsPage

from typing import Dict, List, Union


logger = logging.getLogger(__name__)


class ImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_agilent = 1
    page_csv = 2
    page_perkinelmer = 3
    page_text = 4
    page_thermo = 5
    page_config = 6

    laserImported = QtCore.Signal(Laser)

    def __init__(
        self,
        path: Union[str, Path] = "",
        config: Config = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Import Wizard")

        if isinstance(path, str):  # pragma: no cover
            path = Path(path)

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
        else:  # pragma: no cover
            raise ValueError("Invalid filetype selection.")

        datas = self.field("laserdatas")
        for path, data in zip(paths, datas):
            config = Config(
                spotsize=float(self.field("spotsize")),
                scantime=float(self.field("scantime")),
                speed=float(self.field("speed")),
            )
            self.laserImported.emit(
                Laser(data, config=config, name=path.stem, path=path)
            )
        super().accept()


class FormatPage(QtWidgets.QWizardPage):
    def __init__(
        self,
        text: str,
        page_id_dict: Dict[str, int] = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setTitle("Import Introduction")

        self.page_id_dict = page_id_dict

        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)

        self.radio_agilent = QtWidgets.QRadioButton("&Agilent Batches")
        self.radio_agilent.setChecked(True)
        self.radio_csv = QtWidgets.QRadioButton("&CSV Lines")
        self.radio_numpy = QtWidgets.QRadioButton("&Numpy Archives")
        self.radio_perkinelmer = QtWidgets.QRadioButton("&Perkin-Elmer 'XL'")
        self.radio_text = QtWidgets.QRadioButton("Text and &CSV Images")
        self.radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV")

        format_box = QtWidgets.QGroupBox("File Format")
        layout_format = QtWidgets.QVBoxLayout()
        layout_format.addWidget(self.radio_agilent)
        layout_format.addWidget(self.radio_csv)
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
        self.registerField("numpy", self.radio_numpy)
        self.registerField("text", self.radio_text)
        self.registerField("thermo", self.radio_thermo)

    def nextId(self) -> int:
        if self.page_id_dict is None:  # pragma: no cover
            return super().nextId()

        for field, page_id in self.page_id_dict.items():
            if self.field(field):
                return page_id

        return 0  # pragma: no cover


class ConfigPage(QtWidgets.QWizardPage):
    dataChanged = QtCore.Signal()

    def __init__(self, config: Config, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setTitle("Isotopes and Config")

        self._datas: List[np.ndarray] = []
        self.label_isotopes = QtWidgets.QLabel()
        self.button_isotopes = QtWidgets.QPushButton("Edit Names")
        self.button_isotopes.pressed.connect(self.buttonNamesPressed)

        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setText(str(config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidatorNoZero(0, 1e9, 4))
        self.lineedit_spotsize.textChanged.connect(self.aspectChanged)
        self.lineedit_spotsize.textChanged.connect(self.completeChanged)
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setText(str(config.speed))
        self.lineedit_speed.setValidator(DecimalValidatorNoZero(0, 1e9, 4))
        self.lineedit_speed.textChanged.connect(self.aspectChanged)
        self.lineedit_speed.textChanged.connect(self.completeChanged)
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setText(str(config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidatorNoZero(0, 1e9, 4))
        self.lineedit_scantime.textChanged.connect(self.aspectChanged)
        self.lineedit_scantime.textChanged.connect(self.completeChanged)

        self.lineedit_aspect = QtWidgets.QLineEdit()
        self.lineedit_aspect.setEnabled(False)

        layout_isotopes = QtWidgets.QHBoxLayout()
        layout_isotopes.addWidget(QtWidgets.QLabel("Isotopes:"), 0, QtCore.Qt.AlignLeft)
        layout_isotopes.addWidget(self.label_isotopes, 1)
        layout_isotopes.addWidget(self.button_isotopes, 0, QtCore.Qt.AlignRight)

        config_box = QtWidgets.QGroupBox("Config")
        layout_config = QtWidgets.QFormLayout()
        layout_config.addRow("Spotsize (μm):", self.lineedit_spotsize)
        layout_config.addRow("Speed (μm):", self.lineedit_speed)
        layout_config.addRow("Scantime (s):", self.lineedit_scantime)
        layout_config.addRow("Aspect:", self.lineedit_aspect)
        config_box.setLayout(layout_config)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(
            QtWidgets.QLabel("Edit imported elements and laser configuration."), 0
        )
        layout.addLayout(layout_isotopes)
        layout.addWidget(config_box)

        self.setLayout(layout)

        self.registerField("spotsize", self.lineedit_spotsize)
        self.registerField("speed", self.lineedit_speed)
        self.registerField("scantime", self.lineedit_scantime)

        self.registerField("laserdatas", self, "data_prop")

    def getData(self) -> List[np.ndarray]:
        return self._datas

    def setData(self, datas: List[np.ndarray]) -> None:
        self._datas = datas
        self.dataChanged.emit()

    def getNames(self) -> List[str]:
        data = self.field("laserdatas")[0]
        return data.dtype.names if data is not None else []

    def initializePage(self) -> None:
        data = self.field("laserdatas")[0]
        self.setElidedNames(data.dtype.names)

    def aspectChanged(self) -> None:
        try:
            aspect = (
                float(self.field("speed"))
                * float(self.field("scantime"))
                / float(self.field("spotsize"))
            )
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
        if not self.lineedit_speed.hasAcceptableInput():
            return False
        if not self.lineedit_scantime.hasAcceptableInput():
            return False
        return True

    def setElidedNames(self, names: List[str]) -> None:
        text = ", ".join(name for name in names)
        fm = QtGui.QFontMetrics(self.label_isotopes.font())
        text = fm.elidedText(text, QtCore.Qt.ElideRight, self.label_isotopes.width())
        self.label_isotopes.setText(text)

    def updateNames(self, rename: dict) -> None:
        datas = self.field("laserdatas")
        for i in range(len(datas)):
            remove = [name for name in datas[i].dtype.names if name not in rename]
            datas[i] = rfn.drop_fields(datas[i], remove, usemask=False)
            datas[i] = rfn.rename_fields(datas[i], rename)

        self.setField("laserdatas", datas)
        self.setElidedNames(datas[0].dtype.names)

    data_prop = QtCore.Property("QVariant", getData, setData, notify=dataChanged)
