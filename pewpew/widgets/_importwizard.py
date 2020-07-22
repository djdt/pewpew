import os

# import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.lib import peakfinding
from pew.config import Config
from pew.laser import Laser
from pew.srr import SRRLaser, SRRConfig

from pewpew.actions import qAction, qToolButton
from pewpew.validators import DecimalValidator, DecimalValidatorNoZero
from pewpew.widgets.canvases import BasicCanvas
from pewpew.widgets.ext import MultipleDirDialog

from typing import Dict, List, Tuple, Union


class ImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_files = 1
    page_agilent = 2
    page_text = 3
    page_thermo = 4
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
        self.setPage(ImportWizard.page_agilent, ImportAgilentPage(path, parent=self))
        self.setPage(ImportWizard.page_text, ImportTextPage(path, parent=self))
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
        self.radio_text = QtWidgets.QRadioButton("&Text and CSV images")
        self.radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV")

        format_box = QtWidgets.QGroupBox("File Format")
        layout_format = QtWidgets.QVBoxLayout()
        layout_format.addWidget(self.radio_agilent)
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
        elif self.radio_text.isChecked():
            return ImportWizard.page_text
        elif self.radio_thermo.isChecked():
            return ImportWizard.page_thermo
        return 0


# class ImportFilesWidgetItem(QtWidgets.QListWidgetItem):
#     def __init__(self, path: str = "", parent: QtWidgets.QListWidget = None):
#         super().__init__(path, parent, type=QtWidgets.QListWidgetItem.UserType)

#         # self.id = QtWidgets.QLabel(id)
#         # self.path = QtWidgets.QLineEdit(path)
#         self.action_remove = qAction(
#             "close", "Remove", "Remove this path from the list.", self.remove
#         )
#         self.button_remove = qToolButton(action=self.action_remove)

#         # layout = QtWidgets.QHBoxLayout()
#         # layout.addWidget(self.id, 0)
#         # layout.addWidget(self.path, 1)
#         # layout.addWidget(self.button_close, 0, QtCore.Qt.AlignRight)
#         # self.setLayout(layout)

#     def remove(self) -> None:
#         pass
#         # self.parent().removeItemWidget(self)


# class ImportFilesWidget(QtWidgets.QWidget):
#     def __init__(self, paths: List[str] = None, parent: QtWidgets.QWidget = None):
#         super().__init__(parent)

#         self.list = QtWidgets.QListWidget()

#         self.list.addItem(ImportFilesWidgetItem("/home/tom/0"))
#         self.list.addItem(ImportFilesWidgetItem("/home/tom/0"))
#         self.list.addItem(ImportFilesWidgetItem("/home/tom/0"))

#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(self.list)
#         self.setLayout(layout)

#     # def buttonPathPressed(self) -> QtWidgets.QFileDialog:


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
        # self.layout_path.addWidget(QtWidgets.QLabel("Path:"))
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
        self.setTitle("Agilent Batch")

        self.lineedit_path.setPlaceholderText("Path to batch directory...")
        self.button_path.setText("Open Batch")

        self.combo_dfile_method = QtWidgets.QComboBox()
        self.combo_dfile_method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents
        )
        self.combo_dfile_method.activated.connect(self.updateDataFileCount)

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
        self.registerField("agilent.method", self.combo_dfile_method)
        self.registerField("agilent.names", self.check_name_acq_xml)

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
        self.combo_dfile_method.setCurrentText(current_text)

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


class ImportThermoPage(_ImportOptionsPage):
    def __init__(self, path: str = "", parent: QtWidgets.QWidget = None):
        super().__init__("Thermo iCap Data", [".csv"], path, parent)

        self.radio_columns = QtWidgets.QRadioButton("Exported as columns.")
        self.radio_rows = QtWidgets.QRadioButton("Exported as rows.")


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
