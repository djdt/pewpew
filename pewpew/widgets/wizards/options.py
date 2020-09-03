from PySide2 import QtCore, QtGui, QtWidgets

import os.path

from pew import io

from pewpew.events import DragDropRedirectFilter
from pewpew.widgets.ext import MultipleDirDialog

from typing import List, Tuple


class _OptionsBase(QtWidgets.QGroupBox):
    completeChanged = QtCore.Signal()

    def __init__(
        self, filetype: str, exts: List[str], parent: QtWidgets.QWidget = None
    ):
        super().__init__("Import Options", parent)
        self.filetype = filetype
        self.exts = exts

    def isComplete(self) -> bool:
        return True

    def updateOptions(self, path: str) -> None:
        pass


class AgilentOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Agilent Batch", [".b"], parent)

        self.current_path = ""
        self.actual_datafiles = 0
        self.expected_datafiles = -1

        self.combo_dfile_method = QtWidgets.QComboBox()
        self.combo_dfile_method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents
        )
        self.combo_dfile_method.activated.connect(self.countDatafiles)
        self.combo_dfile_method.activated.connect(self.completeChanged)

        self.lineedit_dfile = QtWidgets.QLineEdit()
        self.lineedit_dfile.setReadOnly(True)

        self.check_name_acq_xml = QtWidgets.QCheckBox(
            "Read names from Acquistion Method."
        )

        dfile_layout = QtWidgets.QFormLayout()
        dfile_layout.addRow("Data File Collection:", self.combo_dfile_method)
        dfile_layout.addRow("Data Files Found:", self.lineedit_dfile)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(dfile_layout, 1)
        layout.addWidget(self.check_name_acq_xml, 0)
        self.setLayout(layout)

    def countDatafiles(self) -> None:
        method = self.combo_dfile_method.currentText()

        if method == "Alphabetical Order":
            datafiles = io.agilent.find_datafiles_alphabetical(self.current_path)
        elif method == "Acquistion Method":
            datafiles = io.agilent.acq_method_xml_read_datafiles(
                self.current_path,
                os.path.join(self.current_path, io.agilent.acq_method_xml_path),
            )
        elif method == "Batch Log CSV":
            datafiles = io.agilent.batch_csv_read_datafiles(
                self.current_path,
                os.path.join(self.current_path, io.agilent.batch_csv_path),
            )
        elif method == "Batch Log XML":
            datafiles = io.agilent.batch_xml_read_datafiles(
                self.current_path,
                os.path.join(self.current_path, io.agilent.batch_xml_path),
            )
        else:
            raise ValueError("Unknown data file collection method.")

        csvs = [
            os.path.join(d, os.path.splitext(os.path.basename(d))[0] + ".csv")
            for d in datafiles
        ]
        self.actual_datafiles = sum([os.path.exists(csv) for csv in csvs])
        self.expected_datafiles = len(datafiles)

        if self.expected_datafiles == 0:
            self.lineedit_dfile.clear()
        else:
            self.lineedit_dfile.setText(
                f"{self.actual_datafiles} ({self.expected_datafiles} expected)"
            )

    def isComplete(self) -> bool:
        return self.actual_datafiles > 0

    def setEnabled(self, enabled: bool) -> None:
        self.combo_dfile_method.setEnabled(enabled)
        self.check_name_acq_xml.setEnabled(enabled)

    def updateOptions(self, path: str) -> None:
        self.current_path = path
        current_text = self.combo_dfile_method.currentText()

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

        self.countDatafiles()


class NumpyOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Numpy Archive", [".npz"], parent)
        self.check_calibration = QtWidgets.QCheckBox("Import calibration.")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_calibration)
        self.setLayout(layout)


class TextOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Text Image", [".csv", ".text", ".txt"], parent)

        self.lineedit_name = QtWidgets.QLineEdit("_Isotope_")
        layout = QtWidgets.QFormLayout()
        layout.addRow("Isotope Name:", self.lineedit_name)

        self.setLayout(layout)

    def isComplete(self) -> bool:
        return self.lineedit_name.text() != ""


class ThermoOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Thermo iCap Data", [".csv"], parent)

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
        self.setLayout(layout)

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

    def setEnabled(self, enabled: bool) -> None:
        self.combo_delimiter.setEnabled(enabled)
        self.combo_decimal.setEnabled(enabled)
        self.radio_rows.setEnabled(enabled)
        self.radio_columns.setEnabled(enabled)

    def updateOptions(self, path: str) -> None:
        delimiter, method, has_analog = self.preprocessFile(path)
        self.combo_delimiter.setCurrentText(delimiter)
        if method == "rows":
            self.radio_rows.setChecked(True)
        elif method == "columns":
            self.radio_columns.setChecked(True)

        if has_analog:
            self.check_use_analog.setEnabled(True)
        else:
            self.check_use_analog.setEnabled(False)
            self.check_use_analog.setChecked(False)


class _PathSelectBase(QtWidgets.QWidget):
    pathChanged = QtCore.Signal()

    def __init__(
        self,
        filetype: str,
        exts: List[str],
        mode: str = "File",
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        if mode not in ["File", "Directory"]:
            raise ValueError("Mode must be one of 'File', 'Directory'.")
        self.mode = mode
        self.filetype = filetype
        self.exts = exts

        self.setAcceptDrops(True)

    @QtCore.Property("QString")
    def path(self) -> str:
        raise NotImplementedError

    @QtCore.Property("QStringList")
    def paths(self) -> List[str]:
        raise NotImplementedError

    def addPath(self, path: str) -> None:
        raise NotImplementedError

    def addPaths(self, paths: List[str]) -> None:
        raise NotImplementedError

    def selectPath(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(self, f"Select {self.mode}", self.currentDir())
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        if self.mode == "File":
            dlg.setNameFilters([self.nameFilter(), "All Files(*)"])
            dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        else:
            dlg.setFileMode(QtWidgets.QFileDialog.Directory)
            dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.addPath)
        dlg.open()
        return dlg

    def selectMultiplePaths(self) -> QtWidgets.QFileDialog:
        if self.mode == "File":
            dlg = QtWidgets.QFileDialog(
                self, "Select Files", os.path.dirname(self.currentDir())
            )
            dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
            dlg.setNameFilters([self.nameFilter(), "All Files(*)"])
            dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        else:
            dlg = MultipleDirDialog(self, "Select Directories", self.currentDir())
        dlg.filesSelected.connect(self.addPaths)
        dlg.open()
        return dlg

    def currentDir(self) -> str:
        return os.path.dirname(self.path)

    def isComplete(self) -> bool:
        if len(self.paths) == 0:
            return False
        return all([self.validPath(path) for path in self.paths])

    def nameFilter(self) -> str:
        return f"{self.filetype}({' '.join(['*' + ext for ext in self.exts])})"

    def validPath(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        name, ext = os.path.splitext(path)
        if not ext.lower() in self.exts:
            return False
        if self.mode == "File":
            return os.path.isfile(path)
        else:
            return os.path.isdir(path)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if self.validPath(path):
                        self.addPath(path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class PathSelectWidget(_PathSelectBase):
    def __init__(
        self,
        path: str,
        filetype: str,
        exts: List[str],
        mode: str = "File",
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(filetype, exts, mode, parent)

        self.lineedit_path = QtWidgets.QLineEdit(path)
        self.lineedit_path.setPlaceholderText(f"Path to {self.mode}...")
        self.lineedit_path.textChanged.connect(self.pathChanged)
        self.lineedit_path.installEventFilter(DragDropRedirectFilter(self))

        self.button_path = QtWidgets.QPushButton(f"Open {self.mode}")
        self.button_path.pressed.connect(self.selectPath)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.lineedit_path, 1)
        layout.addWidget(self.button_path, 0, QtCore.Qt.AlignRight)

        self.setLayout(layout)

    @QtCore.Property("QString")
    def path(self) -> str:
        return self.lineedit_path.text()

    @QtCore.Property("QStringList")
    def paths(self) -> List[str]:
        return [self.lineedit_path.text()]

    def addPath(self, path: str) -> None:
        self.lineedit_path.setText(path)


class MultiplePathSelectWidget(_PathSelectBase):
    def __init__(
        self,
        paths: List[str],
        filetype: str,
        exts: List[str],
        mode: str = "File",
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(filetype, exts, mode, parent)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list.setTextElideMode(QtCore.Qt.ElideLeft)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.list.model().rowsInserted.connect(self.pathChanged)
        self.list.model().rowsRemoved.connect(self.pathChanged)
        self.list.model().rowsMoved.connect(self.pathChanged)

        self.list.addItems(paths)

        text = "Files" if self.mode == "File" else "Directories"
        self.button_path = QtWidgets.QPushButton(f"Open {text}")
        self.button_path.pressed.connect(self.selectMultiplePaths)
        self.button_dir = QtWidgets.QPushButton("Open All...")
        self.button_dir.pressed.connect(self.selectAllInDirectory)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.button_path, 0, QtCore.Qt.AlignRight)
        button_layout.addWidget(self.button_dir, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.setLayout(layout)

    @QtCore.Property("QString")
    def path(self) -> str:
        first = self.list.item(0)
        return first.text() if first is not None else ""

    @QtCore.Property("QStringList")
    def paths(self) -> List[str]:
        return [self.list.item(i).text() for i in range(0, self.list.count())]

    def addPath(self, path: str) -> None:
        self.list.addItem(path)

    def addPaths(self, paths: List[str]) -> None:
        self.list.addItems(paths)

    def addPathsInDirectory(self, directory: str) -> None:
        files = os.listdir(directory)
        files.sort()
        for f in files:
            name, ext = os.path.splitext(f)
            print(ext.lower(), self.exts)
            path = os.path.join(directory, f)
            if ext.lower() in self.exts and self.validPath(path):
                print(path)
                self.list.addItem(path)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in [QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete]:
            items = self.list.selectedItems()
            for item in items:
                self.list.takeItem(self.list.row(item))

    def selectAllInDirectory(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(self, "Select Directory", self.currentDir())
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.addPathsInDirectory)
        dlg.open()
        return dlg
