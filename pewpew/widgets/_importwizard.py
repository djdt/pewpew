import os

# import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.lib import peakfinding
from pew.config import Config
from pew.laser import Laser
from pew.srr import SRRLaser, SRRConfig

from pewpew.validators import DecimalValidator, DecimalValidatorNoZero
from pewpew.widgets.canvases import BasicCanvas
from pewpew.widgets.ext import MultipleDirDialog

from typing import Dict, List, Tuple, Union


class ImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_files = 1
    page_agilent = 2
    page_numpy = 3
    page_text = 4
    page_thermo = 5
    page_config = 6

    def __init__(
        self, path: str = "", config: Config = None, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)

        self.laser: Laser = None
        self.config = config or Config()
        self.path = path

        self.setPage(self.page_format, ImportFormatPage(parent=self))
        # self.setPage(self.page_files, ImportFileAndFormatPage(parent=self))
        self.setPage(ImportWizard.page_agilent, ImportAgilentPage(parent=self))
        # self.setPage(ImportWizard.page_options_csv, ImportOptionsCSVPage(path))
        # self.setPage(ImportWizard.page_options_numpy, ImportOptionsNumpyPage(path))
        # self.setPage(ImportWizard.page_options_thermo, ImportOptionsThermoPage(path))

        # self.setPage(ImportWizard.page_lines, ImportConfigPage(parent=self))


class ImportFormatPage(QtWidgets.QWizardPage):
    formats = ["agilent", "numpy", "text", "thermo"]
    format_exts: Dict[str, Union[str, Tuple[str, ...]]] = {
        ".b": "agilent",
        ".csv": ("csv", "thermo"),
        ".npz": "numpy",
        ".text": "csv",
        ".txt": "csv",
    }

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setTitle("Import Introduction")

        label = QtWidgets.QLabel(
            "The wizard will guide you through importing LA-ICP-MS data and provides a higher level to control than the standard import. To begin select the format of the file(s) being imported."
        )
        label.setWordWrap(True)

        self.radio_agilent = QtWidgets.QRadioButton("&Agilent batch")
        self.radio_numpy = QtWidgets.QRadioButton("&Numpy archive")
        self.radio_text = QtWidgets.QRadioButton("&CSV image")
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

    def initializePage(self) -> None:
        self.radio_agilent.setChecked(True)

    def nextId(self) -> int:
        if self.radio_agilent.isChecked():
            return ImportWizard.page_agilent
        elif self.radio_csv.isChecked():
            return ImportWizard.page_text
        elif self.radio_numpy.isChecked():
            return ImportWizard.page_numpy
        elif self.radio_thermo.isChecked():
            return ImportWizard.page_thermo
        return 0

    # def currentFormat(self) -> Union[str, Tuple[str, ...]]:
    #     base, ext = os.path.splitext(self.lineedit_path.text())
    #     ext = ext.lower().rstrip(".")
    #     print(ext)
    #     return ImportFileAndFormatPage.ext_formats.get(ext, "")


class ImportFileAndFormatPage(QtWidgets.QWizardPage):
    ext_formats: Dict[str, Union[str, Tuple[str, ...]]] = {
        ".b": "agilent",
        ".csv": ("csv", "thermo"),
        ".npz": "numpy",
        ".text": "csv",
        ".txt": "csv",
    }

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setTitle("Select File / Directory")

        self.lineedit_path = QtWidgets.QLineEdit()
        self.lineedit_path.textChanged.connect(self.completeChanged)

        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_file.pressed.connect(self.buttonFilePressed)
        self.button_dir = QtWidgets.QPushButton("Open Directory")
        self.button_dir.pressed.connect(self.buttonDirectoryPressed)

        self.radio_agilent = QtWidgets.QRadioButton("&Agilent batch")
        self.radio_csv = QtWidgets.QRadioButton("&CSV image")
        self.radio_numpy = QtWidgets.QRadioButton("&Numpy archive")
        self.radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV")

        self.registerField("path", self.lineedit_path)

        format_box = QtWidgets.QGroupBox("Import Format")
        layout_format = QtWidgets.QVBoxLayout()
        layout_format.addWidget(self.radio_agilent)
        layout_format.addWidget(self.radio_csv)
        layout_format.addWidget(self.radio_numpy)
        layout_format.addWidget(self.radio_thermo)
        format_box.setLayout(layout_format)

        layout_path = QtWidgets.QHBoxLayout()
        layout_path.addWidget(self.lineedit_path, 1)
        layout_path.addWidget(self.button_file, 0)
        layout_path.addWidget(self.button_dir, 0)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_path)
        layout.addWidget(format_box)
        self.setLayout(layout)

    def initializePage(self) -> None:
        _, ext = os.path.splitext(self.field("path"))
        file_format = ImportFileAndFormatPage.ext_formats.get(ext, "")
        if isinstance(file_format, tuple):
            file_format = file_format[0]

        if file_format == "agilent":
            self.radio_agilent.setChecked(True)
        elif file_format == "csv":
            self.radio_csv.setChecked(True)
        elif file_format == "numpy":
            self.radio_numpy.setChecked(True)

    def cleanupPage(self) -> None:
        self.setField("path", "")

    def buttonFilePressed(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(
            self,
            "Import File",
            os.path.dirname(self.lineedit_path.text()),
            "CSV Documents(*.csv *.txt *.text);;Numpy Archives(*.npz);;All files(*)",
        )
        dlg.selectNameFilter("All files(*)")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.fileSelected.connect(self.fileSelected)
        dlg.open()
        return dlg

    def buttonDirectoryPressed(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(
            self, "Import Directory", os.path.dirname(self.lineedit_path.text())
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.fileSelected)
        dlg.open()
        return dlg

    def fileSelected(self, path: str) -> None:
        self.setField("path", path)
        self.initializePage()

    def isComplete(self) -> bool:
        if self.lineedit_path.text() == "":
            return False
        if not os.path.exists(self.lineedit_path.text()):
            return False
        # if self.currentExt() not in ImportFileAndFormatPage.ext_format.keys():
        #     return False
        return True

    def nextId(self) -> int:
        if self.radio_agilent.isChecked():
            return ImportWizard.page_agilent
        elif self.radio_csv.isChecked():
            return ImportWizard.page_text
        elif self.radio_numpy.isChecked():
            return ImportWizard.page_numpy
        elif self.radio_thermo.isChecked():
            return ImportWizard.page_thermo
        return 0

    # def currentFormat(self) -> Union[str, Tuple[str, ...]]:
    #     base, ext = os.path.splitext(self.lineedit_path.text())
    #     ext = ext.lower().rstrip(".")
    #     print(ext)
    #     return ImportFileAndFormatPage.ext_formats.get(ext, "")


class ImportAgilentPage(QtWidgets.QWizardPage):
    dfile_methods = {
        "AcqMethod.xml": io.agilent.acq_method_xml_path,
        "BatchLog.csv": io.agilent.batch_csv_path,
        "BatchLog.xml": io.agilent.batch_xml_path,
    }

    def __init__(self, path: str = "", parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setTitle("Agilent Batch")

        self.lineedit_path = QtWidgets.QLineEdit(path)
        self.lineedit_path.setPlaceholderText("Path to batch directory...")
        self.lineedit_path.textChanged.connect(self.pathChanged)

        self.button_path = QtWidgets.QPushButton("Open Batch")
        self.button_path.pressed.connect(self.buttonPathPressed)

        self.combo_dfile_method = QtWidgets.QComboBox()
        self.combo_dfile_method.activated.connect(self.updateDataFileCount)

        self.lineedit_dfile = QtWidgets.QLineEdit()
        self.lineedit_dfile.setReadOnly(True)

        self.check_name_acq_xml = QtWidgets.QCheckBox(
            "Read names from Acquistion Method."
        )

        layout_path = QtWidgets.QHBoxLayout()
        layout_path.addWidget(self.lineedit_path, 1)
        layout_path.addWidget(self.button_path, 0)

        dfile_box = QtWidgets.QGroupBox("Data File Collection")
        dfile_layout = QtWidgets.QFormLayout()
        dfile_layout.addRow("Method:", self.combo_dfile_method)
        dfile_layout.addRow("Found:", self.lineedit_dfile)
        dfile_box.setLayout(dfile_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_path)
        layout.addWidget(dfile_box)
        layout.addWidget(self.check_name_acq_xml)
        self.setLayout(layout)

        self.registerField("agilent.path", self.lineedit_path)

    def initializePage(self) -> None:
        self.updateImportOptions()
        self.updateDataFileCount()

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
        self.combo_dfile_method.setCurrentText(current_text)

    def buttonPathPressed(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(
            self, "Import Batch", os.path.dirname(self.lineedit_path.text())
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
        self.completeChanged.emit()

    def pathSelected(self, path: str) -> None:
        self.setField("agilent.path", path)

    def cleanupPage(self) -> None:
        self.combo_dfile_method.clear()

    def isComplete(self) -> bool:
        if not os.path.exists(self.field("agilent.path")):
            return False
        return self.dataFileCount()[1] > 0

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
            return 0, -1
        return len(data_files), sum([os.path.exists(f) for f in data_files])

    def updateDataFileCount(self) -> None:
        expected, actual = self.dataFileCount()
        if (expected, actual) == (0, -1):
            self.lineedit_dfile.clear()
        else:
            self.lineedit_dfile.setText(f"{actual} ({expected} expected)")

    def validPath(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isdir(path)


# class ImportConfigPage(QtWidgets.QWizardPage):
#     def __init__(self, parent: QtWidgets.QWidget = None):
#         super().__init__(parent)

#     # Lines?
#     # Istopes?
#     # Config

if __name__ == "__main__":
    app = QtWidgets.QApplication()
    w = ImportWizard("/home/tom/Downloads/20200630_agar_test_1.b")
    w.show()
    app.exec_()
