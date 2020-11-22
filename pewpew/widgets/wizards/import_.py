import numpy as np
import numpy.lib.recfunctions as rfn
import logging

from pathlib import Path

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib import io
from pewlib.config import Config
from pewlib.laser import Laser

from pewpew.validators import DecimalValidatorNoZero
from pewpew.widgets.dialogs import NameEditDialog
from pewpew.widgets.wizards.options import PathAndOptionsPage

from typing import Dict, List, Tuple, Union


logger = logging.getLogger(__name__)


class ImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_agilent = 1
    page_perkinelmer = 2
    page_text = 3
    page_thermo = 4
    page_config = 5

    laserImported = QtCore.Signal(Laser)

    def __init__(
        self,
        path: Union[str, Path] = "",
        config: Config = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Import Wizard")

        if isinstance(path, str):
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
                "numpy": 0,
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
            path = Path(self.field("agilent.path"))
        if self.field("perkinelmer"):
            path = Path(self.field("perkinelmer.path"))
        elif self.field("text"):
            path = Path(self.field("text.path"))
        elif self.field("thermo"):
            path = Path(self.field("thermo.path"))
        else:
            raise ValueError("Invalid filetype selection.")

        data = self.field("laserdata")
        config = Config(
            spotsize=float(self.field("spotsize")),
            scantime=float(self.field("scantime")),
            speed=float(self.field("speed")),
        )
        self.laserImported.emit(Laser(data, config=config, name=path.stem, path=path))
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
        self.radio_numpy = QtWidgets.QRadioButton("&Numpy Archives")
        self.radio_perkinelmer = QtWidgets.QRadioButton("&Perkin-Elmer 'XL'")
        self.radio_text = QtWidgets.QRadioButton("Text and &CSV Images")
        self.radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV")

        format_box = QtWidgets.QGroupBox("File Format")
        layout_format = QtWidgets.QVBoxLayout()
        layout_format.addWidget(self.radio_agilent)
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
        self.registerField("perkinelmer", self.radio_perkinelmer)
        self.registerField("numpy", self.radio_numpy)
        self.registerField("text", self.radio_text)
        self.registerField("thermo", self.radio_thermo)

    def nextId(self) -> int:
        if self.page_id_dict is None:  # pragma: no cover
            return super().nextId()

        if self.field("agilent"):
            return self.page_id_dict["agilent"]
        if self.field("perkinelmer"):
            return self.page_id_dict["perkinelmer"]
        elif self.field("numpy"):
            return self.page_id_dict["numpy"]
        elif self.field("text"):
            return self.page_id_dict["text"]
        elif self.field("thermo"):
            return self.page_id_dict["thermo"]
        return 0


class ConfigPage(QtWidgets.QWizardPage):
    dataChanged = QtCore.Signal()

    def __init__(self, config: Config, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setTitle("Isotopes and Config")

        self._data: np.ndarray = None
        self.label_isotopes = QtWidgets.QLabel()
        self.button_isotopes = QtWidgets.QPushButton("Edit Names")
        self.button_isotopes.pressed.connect(self.buttonNamesPressed)

        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setText(str(config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidatorNoZero(0, 1e5, 1))
        self.lineedit_spotsize.textChanged.connect(self.aspectChanged)
        self.lineedit_spotsize.textChanged.connect(self.completeChanged)
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setText(str(config.speed))
        self.lineedit_speed.setValidator(DecimalValidatorNoZero(0, 1e5, 1))
        self.lineedit_speed.textChanged.connect(self.aspectChanged)
        self.lineedit_speed.textChanged.connect(self.completeChanged)
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setText(str(config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidatorNoZero(0, 1e5, 4))
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
        layout.addLayout(layout_isotopes)
        layout.addWidget(config_box)

        self.setLayout(layout)

        self.registerField("spotsize", self.lineedit_spotsize)
        self.registerField("speed", self.lineedit_speed)
        self.registerField("scantime", self.lineedit_scantime)

        self.registerField("laserdata", self, "data_prop")

    def getData(self) -> np.ndarray:
        return self._data

    def setData(self, data: np.ndarray) -> None:
        self._data = data
        self.dataChanged.emit()

    def getNames(self) -> List[str]:
        data = self.field("laserdata")
        return data.dtype.names if data is not None else []

    def initializePage(self) -> None:
        if self.field("agilent"):
            data, params = self.readAgilent(Path(self.field("agilent.path")))
        elif self.field("perkinelmer"):
            data, params = self.readPerkinElmer(Path(self.field("perkinelmer.path")))
        elif self.field("text"):
            data, params = self.readText(Path(self.field("text.path")))
        elif self.field("thermo"):
            data, params = self.readThermo(Path(self.field("thermo.path")))

        if "spotsize" in params:
            self.setField("spotsize", f"{params['spotsize']:.4g}")
        if "speed" in params:
            self.setField("speed", f"{params['speed']:.4g}")
        if "scantime" in params:
            self.setField("scantime", f"{params['scantime']:.4g}")

        self.setField("laserdata", data)
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

    def readAgilent(self, path: Path) -> Tuple[np.ndarray, dict]:
        agilent_method = self.field("agilent.method")
        if agilent_method == "Alphabetical Order":
            method = []  # Fallback to alphabetical
        elif agilent_method == "Acquistion Method":
            method = ["acq_method_xml"]
        elif agilent_method == "Batch Log CSV":
            method = ["batch_csv"]
        elif agilent_method == "Batch Log XML":
            method = ["batch_xml"]
        else:
            raise ValueError("Unknown data file collection method.")

        data, params = io.agilent.load(
            path,
            collection_methods=method,
            use_acq_for_names=self.field("agilent.useAcqNames"),
            full=True,
        )
        return data, params

    def readPerkinElmer(self, path: Path) -> Tuple[np.ndarray, dict]:
        data, params = io.perkinelmer.load(path, full=True)
        return data, params

    def readText(self, path: Path) -> Tuple[np.ndarray, dict]:
        data = io.textimage.load(path, name=self.field("text.name"))
        return data, {}

    def readThermo(self, path: Path) -> Tuple[np.ndarray, dict]:
        kwargs = dict(
            delimiter=self.field("thermo.delimiter"),
            comma_decimal=self.field("thermo.decimal") == ",",
        )
        use_analog = self.field("thermo.useAnalog")

        if self.field("thermo.sampleRows"):
            data = io.thermo.icap_csv_rows_read_data(
                path, use_analog=use_analog, **kwargs
            )
            params = io.thermo.icap_csv_rows_read_params(path, **kwargs)
        else:
            data = io.thermo.icap_csv_columns_read_data(
                path, use_analog=use_analog, **kwargs
            )
            params = io.thermo.icap_csv_columns_read_params(path, **kwargs)
        return data, params

    def setElidedNames(self, names: List[str]) -> None:
        text = ", ".join(name for name in names)
        fm = QtGui.QFontMetrics(self.label_isotopes.font())
        text = fm.elidedText(text, QtCore.Qt.ElideRight, self.label_isotopes.width())
        self.label_isotopes.setText(text)

    def updateNames(self, rename: dict) -> None:
        data = self.field("laserdata")
        remove = [name for name in data.dtype.names if name not in rename]
        data = rfn.drop_fields(data, remove, usemask=False)
        data = rfn.rename_fields(data, rename)

        self.setField("laserdata", data)
        self.setElidedNames(data.dtype.names)

    data_prop = QtCore.Property("QVariant", getData, setData, notify=dataChanged)
