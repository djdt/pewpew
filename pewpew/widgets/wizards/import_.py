import os

import numpy as np
import numpy.lib.recfunctions as rfn
import logging

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.config import Config
from pew.laser import Laser

from pewpew.validators import DecimalValidatorNoZero
from pewpew.widgets.dialogs import NameEditDialog
from pewpew.widgets.wizards.importoptions import (
    _ImportOptions,
    AgilentOptions,
    TextOptions,
    ThermoOptions,
)

from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


class ImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_agilent = 1
    page_text = 3
    page_thermo = 4
    page_config = 5

    laserImported = QtCore.Signal(Laser)

    def __init__(
        self, path: str = "", config: Config = None, parent: QtWidgets.QWidget = None
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
                "numpy": 0,
                "text": self.page_text,
                "thermo": self.page_thermo,
            },
            parent=self,
        )
        format_page.radio_numpy.setEnabled(False)
        format_page.radio_numpy.setVisible(False)

        self.setPage(self.page_format, format_page)
        self.setPage(self.page_agilent, AgilentPage(path, parent=self))
        self.setPage(self.page_text, TextPage(path, parent=self))
        self.setPage(self.page_thermo, ThermoPage(path, parent=self))

        self.setPage(self.page_config, ConfigPage(config, parent=self))

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
        self.radio_text = QtWidgets.QRadioButton("Text and &CSV Images")
        self.radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV")

        format_box = QtWidgets.QGroupBox("File Format")
        layout_format = QtWidgets.QVBoxLayout()
        layout_format.addWidget(self.radio_agilent)
        layout_format.addWidget(self.radio_numpy)
        layout_format.addWidget(self.radio_text)
        layout_format.addWidget(self.radio_thermo)
        format_box.setLayout(layout_format)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(format_box)
        self.setLayout(layout)

        self.registerField("agilent", self.radio_agilent)
        self.registerField("numpy", self.radio_numpy)
        self.registerField("text", self.radio_text)
        self.registerField("thermo", self.radio_thermo)

    def nextId(self) -> int:
        if self.page_id_dict is None:  # pragma: no cover
            return super().nextId()

        if self.field("agilent"):
            return self.page_id_dict["agilent"]
        elif self.field("numpy"):
            return self.page_id_dict["numpy"]
        elif self.field("text"):
            return self.page_id_dict["text"]
        elif self.field("thermo"):
            return self.page_id_dict["thermo"]
        return 0


class _PageOptions(QtWidgets.QWizardPage):
    def __init__(
        self, options: _ImportOptions, path: str = "", parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setTitle(options.filetype + " Import")

        self.lineedit_path = QtWidgets.QLineEdit(path)
        self.lineedit_path.setPlaceholderText("Path to file...")
        self.lineedit_path.textChanged.connect(self.pathChanged)

        self.button_path = QtWidgets.QPushButton("Open File")
        self.button_path.pressed.connect(self.buttonPathPressed)

        layout_path = QtWidgets.QHBoxLayout()
        layout_path.addWidget(self.lineedit_path, 1)
        layout_path.addWidget(self.button_path, 0, QtCore.Qt.AlignRight)

        self.options = options
        self.options.completeChanged.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_path, 0)
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
        if not self.validPath(self.lineedit_path.text()):
            return False
        return self.options.isComplete()

    def nameFilter(self) -> str:
        return f"{self.options.filetype}({' '.join(['*' + ext for ext in self.options.exts])})"

    def nextId(self) -> int:
        return ImportWizard.page_config

    def pathChanged(self, path: str) -> None:
        self.completeChanged.emit()

    def pathSelected(self, path: str) -> None:
        raise NotImplementedError

    def validPath(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isfile(path)


class AgilentPage(_PageOptions):
    def __init__(self, path: str = "", parent: QtWidgets.QWidget = None):
        super().__init__(AgilentOptions(), path, parent)

        self.lineedit_path.setPlaceholderText("Path to batch directory...")
        self.button_path.setText("Open Batch")

        self.registerField("agilent.path", self.lineedit_path)
        self.registerField(
            "agilent.method",
            self.options.combo_dfile_method,
            "currentText",
            "currentTextChanged",
        )
        self.registerField("agilent.acqNames", self.options.check_name_acq_xml)

    def initializePage(self) -> None:
        self.pathChanged(self.field("agilent.path"))

    def buttonPathPressed(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(
            self, "Select Batch", os.path.dirname(self.lineedit_path.text())
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.pathSelected)
        dlg.open()
        return dlg

    def pathChanged(self, path: str) -> None:
        if self.validPath(path):
            self.options.setEnabled(True)
            self.options.updateOptions(path)
        else:
            self.options.actual_datafiles = 0
            self.options.setEnabled(False)
        super().pathChanged(path)

    def pathSelected(self, path: str) -> None:
        self.setField("agilent.path", path)

    def validPath(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isdir(path)


class TextPage(_PageOptions):
    def __init__(self, path: str = "", parent: QtWidgets.QWidget = None):
        super().__init__(TextOptions(), path, parent)
        self.registerField("text.path", self.lineedit_path)
        self.registerField("text.name", self.options.lineedit_name)

    def pathSelected(self, path: str) -> None:
        self.setField("text.path", path)


class ThermoPage(_PageOptions):
    def __init__(self, path: str = "", parent: QtWidgets.QWidget = None):
        super().__init__(ThermoOptions(), path, parent)

        self.registerField("thermo.path", self.lineedit_path)
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
        self.pathChanged(self.field("thermo.path"))

    def pathChanged(self, path: str) -> None:
        if self.validPath(path):
            self.options.setEnabled(True)
            self.options.updateOptions(path)
        else:
            self.options.setEnabled(False)
        super().pathChanged(path)

    def pathSelected(self, path: str) -> None:
        self.setField("thermo.path", path)


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
            data, params = self.readAgilent(self.field("agilent.path"))
        elif self.field("text"):
            data, params = self.readText(self.field("text.path"))
        elif self.field("thermo"):
            data, params = self.readThermo(self.field("thermo.path"))

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
        except ZeroDivisionError:
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

    def readAgilent(self, path: str) -> Tuple[np.ndarray, dict]:
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

        data, params = io.agilent.load(
            path,
            collection_methods=method,
            use_acq_for_names=self.field("agilent.acqNames"),
            full=True,
        )
        return data, params

    def readText(self, path: str) -> Tuple[np.ndarray, dict]:
        data = io.csv.load(path, isotope=self.field("text.name"))
        return data, {}

    def readThermo(self, path: str) -> Tuple[np.ndarray, dict]:
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
