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
    page_files = 0
    page_options_agilent = 1
    page_options_csv = 2
    page_options_numpy = 3
    page_options_thermo = 4

    def __init__(
        self, path: str = "", config: Config = None, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)

        self.laser: Laser = None
        self.config = config or Config()

        self.setPage(self.page_files, ImportFileAndFormatPage(path, parent=self))
        self.setPage(
            ImportWizard.page_options_agilent, ImportOptionsAgilentPage(parent=self)
        )
        # self.setPage(ImportWizard.page_options_csv, ImportOptionsCSVPage(path))
        # self.setPage(ImportWizard.page_options_numpy, ImportOptionsNumpyPage(path))
        # self.setPage(ImportWizard.page_options_thermo, ImportOptionsThermoPage(path))

        self.setStartId(self.page_files)
        if path != "":
            self.setStartId(self.nextId())


class ImportFileAndFormatPage(QtWidgets.QWizardPage):
    ext_formats: Dict[str, Union[str, Tuple[str, ...]]] = {
        ".b": "agilent",
        ".csv": ("csv", "thermo"),
        ".npz": "numpy",
        ".text": "csv",
        ".txt": "csv",
    }

    def __init__(self, path: str, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setTitle("Select File / Directory")

        self.lineedit_path = QtWidgets.QLineEdit(path)
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
        self.lineedit_path.setText(path)

        _, ext = os.path.splitext(self.lineedit_path.text())
        file_format = ImportFileAndFormatPage.ext_formats.get(ext, "")
        if isinstance(file_format, tuple):
            file_format = file_format[0]

        if file_format == "agilent":
            self.radio_agilent.setChecked(True)
        elif file_format == "csv":
            self.radio_csv.setChecked(True)
        elif file_format == "numpy":
            self.radio_numpy.setChecked(True)

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
            return ImportWizard.page_options_agilent
        elif self.radio_csv.isChecked():
            return ImportWizard.page_options_csv
        elif self.radio_numpy.isChecked():
            return ImportWizard.page_options_numpy
        elif self.radio_thermo.isChecked():
            return ImportWizard.page_options_thermo
        return 0

    # def currentFormat(self) -> Union[str, Tuple[str, ...]]:
    #     base, ext = os.path.splitext(self.lineedit_path.text())
    #     ext = ext.lower().rstrip(".")
    #     print(ext)
    #     return ImportFileAndFormatPage.ext_formats.get(ext, "")


class ImportOptionsAgilentPage(QtWidgets.QWizardPage):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setTitle("Options for Agilent Batches")

        path: str = self.field("path")

        self.combo_dfile_method = QtWidgets.QComboBox()
        self.combo_dfile_method.addItem("Alphabetical Order")

        if os.path.exists(os.path.join(path, io.agilent.acq_method_path)):
            self.combo_dfile_method.addItem("AcqMethod.xml")
        if os.path.exists(os.path.join(path, io.agilent.batch_csv_path)):
            self.combo_dfile_method.addItem("BatchLog.csv")
        if os.path.exists(os.path.join(path, io.agilent.batch_xml_path)):
            self.combo_dfile_method.addItem("BatchLog.xml")

    def dataFileCount(self) -> int:
        path = self.field("path")
        method = self.combo_dfile_method.currentText()

        if method == "Alphabetical Order":
            return len(io.agilent.find_datafiles(path))
        elif method == "AcqMethod.xml":
            return len(
                io.agilent.acq_method_read_datafiles(
                    os.path.join(path, io.agilent.acq_method_path)
                )
            )
        elif method == "BatchLog.csv":
            return len(
                io.agilent.batch_csv_read_datafiles(
                    os.path.join(path, io.agilent.batch_csv_path)
                )
            )
        elif method == "BatchLog.xml":
            return len(
                io.agilent.batch_xml_read_datafiles(
                    os.path.join(path, io.agilent.batch_xml_path)
                )
            )
        else:
            return 0

        # Check for batch log / acq meth

        # Import options files

        # if label is None:
        #     label = QtWidgets.QLabel()
        # label.setWordWrap(True)

        # main_layout = QtWidgets.QVBoxLayout()
        # main_layout.addWidget(label)
        # main_layout.addWidget(mode_box)
        # self.setLayout(main_layout)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    w = ImportWizard()
    w.show()
    app.exec_()
