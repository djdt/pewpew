import os
import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.lib import peakfinding
from pew.config import Config
from pew.laser import Laser
from pew.srr import SRRLaser, SRRConfig

from pewpew.validators import DecimalValidator, DecimalValidatorNoZero
from pewpew.widgets.canvases import BasicCanvas
from pewpew.widgets.ext import MultipleDirDialog

from typing import List, Tuple, Union


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

        self.addPage(ImportFileAndFormatPage(path))

        self.addPage(ImportOptionsAgilentPage(path))
        self.addPage(ImportOptionsCSVPage(path))
        self.addPage(ImportOptionsNumpyPage(path))
        self.addPage(ImportOptionsThermoPage(path))

        if path is not None:
            self.setStartId(self.nextId())


class ImportFileAndFormatPage(QtWidgets.QWizardPage):
    ext_format = {
        "b": "agilent",
        "csv": ("csv", "thermo"),
        "npz": "numpy",
        "text": "csv",
        "txt": "csv",
    }

    def __init__(self, path: str, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.path = path

        self.lineedit_file = QtWidgets.QLineEdit(self.path)
        self.lineedit_file.textEdited.connect(self.completeChanged)
        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_dir = QtWidgets.QPushButton("Open Directory")

        self.radio_agilent = QtWidgets.QRadioButton("&Agilent batch")
        self.radio_csv = QtWidgets.QRadioButton("&CSV document")
        self.radio_numpy = QtWidgets.QRadioButton("&Numpy archive")
        self.radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV")

        format_box = QtWidgets.QGroupBox("Format")
        layout_format = QtWidgets.QVBoxLayout()
        layout_format.addWidget(self.radio_agilent)
        layout_format.addWidget(self.radio_csv)
        layout_format.addWidget(self.radio_numpy)
        layout_format.addWidget(self.radio_thermo)
        format_box.setLayout(layout_format)

    def isComplete(self) -> bool:
        if self.path == "":
            return False
        if not os.path.exists(self.path):
            return False
        if self.currentExt() not in ImportFileAndFormatPage.ext_format.keys():
            return False
        return True

    def buttonFilePressed(self) -> None:
        dlg = QtWidgets.QFileDialog

    def fileSelected(self, path: str) -> None:
        self.path = path
        self.lineedit_file.setText(path)

        file_format = self.currentFormat()
        if isinstance(file_format, tuple):
            file_format = file_format[0]

        if file_format == "agilent":
            self.radio_agilent.setChecked(True)
        elif file_format == "csv":
            self.radio_csv.setChecked(True)
        elif file_format == "numpy":
            self.radio_numpy.setChecked(True)

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

    def currentFormat(self) -> Union[str, Tuple[str, ...]]:
        base, ext = os.path.splitext(self.path)
        ext = ext.lower().rstrip(".")
        return ImportOptionsAgilent.ext_format.get(ext, "")


class ImportOptionsAgilent(QtWidgets.QWizardPage):
    def __init__(self, path: str, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.setTitle("Overview")
        if label is None:
            label = QtWidgets.QLabel()
        label.setWordWrap(True)

        mode_box = QtWidgets.QGroupBox("Data type", self)

        radio_numpy = QtWidgets.QRadioButton("&Numpy archives", self)
        radio_agilent = QtWidgets.QRadioButton("&Agilent batches", self)
        radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV exports", self)
        radio_csv = QtWidgets.QRadioButton("&CSV documents", self)
        radio_numpy.setChecked(True)

        self.registerField("radio_numpy", radio_numpy)
        self.registerField("radio_agilent", radio_agilent)
        self.registerField("radio_thermo", radio_thermo)

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(radio_numpy)
        box_layout.addWidget(radio_agilent)
        box_layout.addWidget(radio_thermo)
        mode_box.setLayout(box_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(label)
        main_layout.addWidget(mode_box)
        self.setLayout(main_layout)
