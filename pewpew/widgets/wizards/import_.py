import os

import numpy as np
import numpy.lib.recfunctions as rfn
import logging

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.config import Config
from pew.laser import Laser

from pewpew.widgets.dialogs import NameEditDialog
from pewpew.validators import DecimalValidator

from typing import Dict, List, Tuple, Union


logger = logging.getLogger(__name__)


class ImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_files = 1
    page_agilent = 2
    page_text = 3
    page_thermo = 4
    page_config = 6

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

        format_page = ImportFormatPage(
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

        self.setPage(self.page_format, format_page)
        self.setPage(self.page_agilent, ImportAgilentPage(path, parent=self))
        self.setPage(self.page_text, ImportTextPage(path, parent=self))
        self.setPage(self.page_thermo, ImportThermoPage(path, parent=self))

        self.setPage(self.page_config, ImportConfigPage(config, parent=self))

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


class ImportFormatPage(QtWidgets.QWizardPage):
    formats = ["agilent", "numpy", "text", "thermo"]
    format_exts: Dict[str, Union[str, Tuple[str, ...]]] = {
        ".b": "agilent",
        ".csv": ("csv", "thermo"),
        ".npz": "numpy",
        ".text": "csv",
        ".txt": "csv",
    }

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


class _ImportOptionsPage(QtWidgets.QWizardPage):
    def __init__(
        self,
        file_type: str,
        file_exts: List[str],
        path: str = "",
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setTitle(file_type + " Import")

        self.file_type = file_type
        self.file_exts = file_exts

        self.lineedit_path = QtWidgets.QLineEdit(path)
        self.lineedit_path.setPlaceholderText("Path to file...")
        self.lineedit_path.textChanged.connect(self.pathChanged)

        self.button_path = QtWidgets.QPushButton("Open File")
        self.button_path.pressed.connect(self.buttonPathPressed)

        layout_path = QtWidgets.QHBoxLayout()
        layout_path.addWidget(self.lineedit_path, 1)
        layout_path.addWidget(self.button_path, 0, QtCore.Qt.AlignRight)

        self.options_box = QtWidgets.QGroupBox("Import Options")

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_path, 0)
        layout.addWidget(self.options_box, 1)
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

    def nameFilter(self) -> str:
        return f"{self.file_type}({' '.join(['*' + ext for ext in self.file_exts])})"

    def nextId(self) -> int:
        return ImportWizard.page_config

    def pathChanged(self, path: str) -> None:
        self.completeChanged.emit()

    def pathSelected(self, path: str) -> None:
        raise NotImplementedError

    def validPath(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isfile(path)


class ImportAgilentPage(_ImportOptionsPage):
    dfile_methods = {
        "AcqMethod.xml": io.agilent.acq_method_xml_path,
        "BatchLog.csv": io.agilent.batch_csv_path,
        "BatchLog.xml": io.agilent.batch_xml_path,
    }

    def __init__(self, path: str = "", parent: QtWidgets.QWidget = None):
        super().__init__("Agilent Batch", ["*.b"], path, parent)

        self.lineedit_path.setPlaceholderText("Path to batch directory...")
        self.button_path.setText("Open Batch")

        self.combo_dfile_method = QtWidgets.QComboBox()
        self.combo_dfile_method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents
        )
        self.combo_dfile_method.activated.connect(self.updateDataFileCount)
        self.combo_dfile_method.activated.connect(self.completeChanged)

        self.lineedit_dfile = QtWidgets.QLineEdit()
        self.lineedit_dfile.setReadOnly(True)

        self.check_name_acq_xml = QtWidgets.QCheckBox(
            "Read names from Acquistion Method."
        )

        dfile_layout = QtWidgets.QFormLayout()
        dfile_layout.addRow("Data File Collection:", self.combo_dfile_method)
        dfile_layout.addRow("Data Files Found:", self.lineedit_dfile)

        layout_options = QtWidgets.QVBoxLayout()
        layout_options.addLayout(dfile_layout, 1)
        layout_options.addWidget(self.check_name_acq_xml, 0)
        self.options_box.setLayout(layout_options)

        self.registerField("agilent.path", self.lineedit_path)
        self.registerField(
            "agilent.method",
            self.combo_dfile_method,
            "currentText",
            "currentTextChanged",
        )
        self.registerField("agilent.acqNames", self.check_name_acq_xml)

    def dataFileCount(self) -> Tuple[int, int]:
        path = self.field("agilent.path")
        if not self.validPath(path):
            return 0, -1

        method = self.combo_dfile_method.currentText()

        if method == "Alphabetical Order":
            data_files = io.agilent.find_datafiles_alphabetical(path)
        elif method == "Acquistion Method":
            data_files = io.agilent.acq_method_xml_read_datafiles(
                path, os.path.join(path, io.agilent.acq_method_xml_path)
            )
        elif method == "Batch Log CSV":
            data_files = io.agilent.batch_csv_read_datafiles(
                path, os.path.join(path, io.agilent.batch_csv_path)
            )
        elif method == "Batch Log XML":
            data_files = io.agilent.batch_xml_read_datafiles(
                path, os.path.join(path, io.agilent.batch_xml_path)
            )
        else:
            raise ValueError("Unknown data file collection method.")

        csvs = [
            os.path.join(d, os.path.splitext(os.path.basename(d))[0] + ".csv")
            for d in data_files
        ]
        return len(data_files), sum([os.path.exists(csv) for csv in csvs])

    def initializePage(self) -> None:
        self.updateImportOptions()
        self.updateDataFileCount()

    def isComplete(self) -> bool:
        if not self.validPath(self.field("agilent.path")):
            return False
        return self.dataFileCount()[1] > 0

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
        self.updateImportOptions()
        self.updateDataFileCount()
        super().pathChanged(path)

    def pathSelected(self, path: str) -> None:
        self.setField("agilent.path", path)

    def updateDataFileCount(self) -> None:
        expected, actual = self.dataFileCount()
        if (expected, actual) == (0, -1):
            self.lineedit_dfile.clear()
        else:
            self.lineedit_dfile.setText(f"{actual} ({expected} expected)")

    def updateImportOptions(self) -> None:
        path: str = self.field("agilent.path")

        if not self.validPath(path):
            self.check_name_acq_xml.setEnabled(False)
            self.combo_dfile_method.setEnabled(False)
            return

        current_text = self.combo_dfile_method.currentText()

        self.combo_dfile_method.setEnabled(True)
        self.combo_dfile_method.clear()

        self.combo_dfile_method.addItem("Alphabetical Order")
        if os.path.exists(os.path.join(path, io.agilent.acq_method_xml_path)):
            self.combo_dfile_method.addItem("Acquistion Method")
            self.check_name_acq_xml.setEnabled(True)
        else:
            self.check_name_acq_xml.setEnabled(False)
        if os.path.exists(os.path.join(path, io.agilent.batch_csv_path)):
            self.combo_dfile_method.addItem("Batch Log CSV")
        if os.path.exists(os.path.join(path, io.agilent.batch_xml_path)):
            self.combo_dfile_method.addItem("Batch Log XML")

        # Restore the last method if available
        if current_text != "":
            self.combo_dfile_method.setCurrentText(current_text)
        else:
            self.combo_dfile_method.setCurrentIndex(self.combo_dfile_method.count() - 1)

    def validPath(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isdir(path)


class ImportTextPage(_ImportOptionsPage):
    def __init__(self, path: str = "", parent: QtWidgets.QWidget = None):
        super().__init__("Text Image", [".csv", ".text", ".txt"], path, parent)

        self.lineedit_name = QtWidgets.QLineEdit("_Isotope_")

        layout_name = QtWidgets.QFormLayout()
        layout_name.addRow("Isotope Name:", self.lineedit_name)

        self.options_box.setLayout(layout_name)

        self.registerField("text.path", self.lineedit_path)
        self.registerField("text.name", self.lineedit_name)

    def isComplete(self) -> bool:
        if not self.validPath(self.field("text.path")):
            return False
        if self.lineedit_name.text() == "":
            return False
        return True

    def pathSelected(self, path: str) -> None:
        self.setField("text.path", path)


class ImportThermoPage(_ImportOptionsPage):
    def __init__(self, path: str = "", parent: QtWidgets.QWidget = None):
        super().__init__("Thermo iCap Data", [".csv"], path, parent)

        self.radio_columns = QtWidgets.QRadioButton("Samples in columns.")
        self.radio_rows = QtWidgets.QRadioButton("Samples in rows.")

        self.combo_delimiter = QtWidgets.QComboBox()
        self.combo_delimiter.addItems([",", ";"])
        self.combo_decimal = QtWidgets.QComboBox()
        self.combo_decimal.addItems([".", ","])

        self.check_use_analog = QtWidgets.QCheckBox(
            "Use exported analog readings instead of counts."
        )

        layout_radio = QtWidgets.QVBoxLayout()
        layout_radio.addWidget(self.radio_columns)
        layout_radio.addWidget(self.radio_rows)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Export format:", layout_radio)
        layout.addRow("Delimiter:", self.combo_delimiter)
        layout.addRow("Decimal:", self.combo_decimal)
        layout.addRow(self.check_use_analog)
        self.options_box.setLayout(layout)

        self.registerField("thermo.path", self.lineedit_path)
        self.registerField("thermo.sampleColumns", self.radio_columns)
        self.registerField("thermo.sampleRows", self.radio_rows)
        self.registerField(
            "thermo.delimiter",
            self.combo_delimiter,
            "currentText",
            "currentTextChanged",
        )
        self.registerField(
            "thermo.decimal", self.combo_decimal, "currentText", "currentTextChanged"
        )
        self.registerField("thermo.useAnalog", self.check_use_analog)

    def initializePage(self) -> None:
        self.radio_rows.setChecked(True)
        self.combo_decimal.setCurrentText(".")
        self.updateImportOptions()

    def isComplete(self) -> bool:
        if not self.validPath(self.field("thermo.path")):
            return False
        return True

    def updateImportOptions(self) -> None:
        path = self.field("thermo.path")

        if self.validPath(path):
            delimiter, method, has_analog = self.preprocessFile(path)
            self.combo_delimiter.setCurrentText(delimiter)
            if method == "rows":
                self.radio_rows.setChecked(True)
            elif method == "columns":
                self.radio_columns.setChecked(True)

            if not has_analog:
                self.check_use_analog.setChecked(False)

            self.combo_delimiter.setEnabled(True)
            self.combo_decimal.setEnabled(True)
            self.check_use_analog.setEnabled(has_analog)
            self.radio_rows.setEnabled(True)
            self.radio_columns.setEnabled(True)
        else:
            self.combo_delimiter.setEnabled(False)
            self.combo_decimal.setEnabled(False)
            self.radio_rows.setEnabled(False)
            self.radio_columns.setEnabled(False)

    def preprocessFile(self, path: str) -> Tuple[str, str, bool]:
        method = "unknown"
        has_analog = False
        with open(path, "r", encoding="utf-8-sig") as fp:
            lines = [next(fp) for i in range(3)]
            delimiter = lines[0][0]
            if "MainRuns" in lines[0]:
                method = "rows"
            elif "MainRuns" in lines[2]:
                method = "columns"
            for line in fp:
                if "Analog" in line:
                    has_analog = True
                    break
            return delimiter, method, has_analog

    def pathChanged(self, path: str) -> None:
        self.updateImportOptions()
        super().pathChanged(path)

    def pathSelected(self, path: str) -> None:
        self.setField("thermo.path", path)


class ImportConfigPage(QtWidgets.QWizardPage):
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
        self.lineedit_spotsize.setValidator(DecimalValidator(0, 1e5, 1))
        self.lineedit_spotsize.textChanged.connect(self.aspectChanged)
        self.lineedit_spotsize.textChanged.connect(self.completeChanged)
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setText(str(config.speed))
        self.lineedit_speed.setValidator(DecimalValidator(0, 1e5, 1))
        self.lineedit_speed.textChanged.connect(self.aspectChanged)
        self.lineedit_speed.textChanged.connect(self.completeChanged)
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setText(str(config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidator(0, 1e5, 4))
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

    def initializePage(self) -> None:
        if self.field("agilent"):
            data, params = self.readAgilent()
        elif self.field("text"):
            data, params = self.readText()
        elif self.field("thermo"):
            data, params = self.readThermo()

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
        except ValueError:
            self.lineedit_aspect.clear()

    def buttonNamesPressed(self) -> QtWidgets.QDialog:
        data = self.field("laserdata")
        dlg = NameEditDialog(data.dtype.names, parent=self)
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

    def readAgilent(self) -> Tuple[np.ndarray, dict]:
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
            raise ValueError("Unkonw data file collection method.")

        data, params = io.agilent.load(
            self.field("agilent.path"),
            collection_methods=method,
            use_acq_for_names=self.field("agilent.acqNames"),
            full=True,
        )
        return data, params

    def readText(self) -> Tuple[np.ndarray, dict]:
        data = io.csv.load(self.field("text.path"), isotope=self.field("text.name"))
        return data, {}

    def readThermo(self) -> Tuple[np.ndarray, dict]:
        kwargs = dict(
            delimiter=self.field("thermo.delimiter"),
            comma_decimal=self.field("thermo.decimal") == ",",
        )
        use_analog = self.field("thermo.useAnalog")

        if self.field("thermo.sampleRows"):
            data = io.thermo.icap_csv_rows_read_data(
                self.field("thermo.path"), use_analog=use_analog, **kwargs
            )
            params = io.thermo.icap_csv_rows_read_params(
                self.field("thermo.path"), **kwargs
            )
        else:
            data = io.thermo.icap_csv_columns_read_data(
                self.field("thermo.path"), use_analog=use_analog, **kwargs
            )
            params = io.thermo.icap_csv_columns_read_params(
                self.field("thermo.path"), **kwargs
            )
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
