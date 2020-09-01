import os

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.config import Config
from pew.srr import SRRLaser, SRRConfig

from pewpew.actions import qAction, qToolButton
from pewpew.validators import DecimalValidator, DecimalValidatorNoZero

from pewpew.widgets.wizards.import_ import ImportFormatPage
from pewpew.widgets.wizards.importoptions import (
    _ImportOptions,
    ImportOptionsAgilent,
    ImportOptionsText,
    ImportOptionsThermo,
)

from typing import List


class SRRImportWizard(QtWidgets.QWizard):
    page_format = 0
    page_files = 1
    page_agilent = 2
    page_numpy = 3
    page_text = 4
    page_thermo = 5
    page_config = 6

    laserImported = QtCore.Signal(SRRLaser)

    def __init__(
        self,
        paths: List[str] = [],
        config: Config = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SRR Import Wizard")

        config = config or SRRConfig()

        overview = (
            "The wizard will guide you through importing LA-ICP-MS data "
            "and provides a higher level to control than the standard import. "
            "To begin select the format of the file being imported."
        )

        format_page = ImportFormatPage(
            overview,
            page_id_dict={
                "agilent": self.page_agilent,
                "numpy": self.page_numpy,
                "text": self.page_text,
                "thermo": self.page_thermo,
            },
            parent=self,
        )

        self.setPage(self.page_format, format_page)
        self.setPage(self.page_agilent, SRRImportAgilentPage(paths, parent=self))
        self.setPage(self.page_numpy, SRRImportNumpyPage(paths, parent=self))
        self.setPage(self.page_text, SRRImportTextPage(paths, parent=self))
        self.setPage(self.page_thermo, SRRImportThermoPage(paths, parent=self))

        self.setPage(self.page_config, SRRImportConfigPage(config, parent=self))

    def accept(self) -> None:
        if self.field("agilent"):
            path = self.field("agilent.path")
        elif self.field("text"):
            path = self.field("text.path")
        elif self.field("thermo"):
            path = self.field("thermo.path")
        else:
            raise ValueError("Invalid filetype selection.")

        # data = self.field("laserdata")
        # config = Config(
        #     spotsize=float(self.field("spotsize")),
        #     scantime=float(self.field("scantime")),
        #     speed=float(self.field("speed")),
        # )
        # base, ext = os.path.splitext(path)
        # self.laserImported.emit(
        #     SRRLaser(data, config=config, path=path, name=os.path.basename(base))
        # )
        # super().accept()

    def accept(self) -> None:
        self.config.spotsize = float(self.field("spotsize"))
        self.config.speed = float(self.field("speed"))
        self.config.scantime = float(self.field("scantime"))
        self.config.warmup = float(self.field("warmup"))

        subpixel_width = self.field("subpixel_width")
        self.config.set_equal_subpixel_offsets(subpixel_width)

        paths = self.field("paths")
        layers = []

        if self.field("numpy"):
            for path in paths:
                lds = io.npz.load(path)
                if len(lds) > 1:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Import Error",
                        f'Archive "{os.path.basename(path)}" '
                        "contains more than one image.",
                    )
                    return
                layers.append(lds[0].get())
        elif self.field("agilent"):
            for path in paths:
                layers.append(io.agilent.load(path))
        elif self.field("thermo"):
            for path in paths:
                layers.append(io.thermo.load(path))

        self.laserImported.emit(
            SRRLaser(
                layers,
                config=self.config,
                name=os.path.splitext(os.path.basename(paths[0]))[0],
                path=paths[0],
            )
        )
        super().accept()


class _SRRImportOptionsPage(QtWidgets.QWizardPage):
    def __init__(
        self,
        options: _ImportOptions,
        paths: List[str] = [],
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setTitle(options.filetype + " Import")

        self.paths = PathsSelectionWidget(paths, options.filetype, options.exts)
        self.paths.pathsChanged.connect(self.pathsChanged)

        self.options = options
        self.options.completeChanged.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.path, 0)
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
        paths = self.paths.paths
        if len(paths) < 2:
            return False
        for path in paths:
            if not self.validPath(path):
                return False
        return self.options.isComplete()

    def nameFilter(self) -> str:
        return f"{self.options.filetype}({' '.join(['*' + ext for ext in self.options.exts])})"

    def nextId(self) -> int:
        return SRRImportWizard.page_config

    def pathsChanged(self) -> None:
        self.completeChanged.emit()

    def validPath(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isfile(path)


class SRRImportAgilentPage(_SRRImportOptionsPage):
    def __init__(self, paths: List[str] = [], parent: QtWidgets.QWidget = None):
        super().__init__(ImportOptionsAgilent(), paths, parent)
        self.paths.uses_directories = True

        self.registerField("agilent.paths", self.paths.paths)
        self.registerField(
            "agilent.method",
            self.options.combo_dfile_method,
            "currentText",
            "currentTextChanged",
        )
        self.registerField("agilent.acqNames", self.options.check_name_acq_xml)

    def initializePage(self) -> None:
        self.pathsChanged()

    def pathsChanged(self) -> None:
        paths = self.field("agilent.paths")
        path = paths[0] if len(paths) > 0 else ""
        if self.validPath(path):
            self.options.setEnabled(True)
            self.options.updateOptions(path)
        else:
            self.options.actual_datafiles = 0
            self.options.setEnabled(False)
        super().pathsChanged()

    def validPath(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isdir(path)


class SRRImportTextPage(_SRRImportOptionsPage):
    def __init__(self, paths: List[str] = [], parent: QtWidgets.QWidget = None):
        super().__init__(ImportOptionsText(), paths, parent)
        self.registerField("text.paths", self.paths.paths)
        self.registerField("text.name", self.options.lineedit_name)


class SRRImportThermoPage(_SRRImportOptionsPage):
    def __init__(self, paths: List[str] = [], parent: QtWidgets.QWidget = None):
        super().__init__(ImportOptionsThermo(), paths, parent)

        self.registerField("thermo.paths", self.paths.paths)
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
        self.pathsChanged()

    def pathChanged(self, path: str) -> None:
        paths = self.field("thermo.paths")
        path = paths[0] if len(paths) > 0 else ""
        if self.validPath(path):
            self.options.setEnabled(True)
            self.options.updateOptions(path)
        else:
            self.options.setEnabled(False)
        super().pathChanged(path)


class SRRConfigPage(QtWidgets.QWizardPage):
    def __init__(self, config: SRRConfig, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setText(str(config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidatorNoZero(0, 1e3, 4))
        self.lineedit_spotsize.textEdited.connect(self.completeChanged)

        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setText(str(config.speed))
        self.lineedit_speed.setValidator(DecimalValidatorNoZero(0, 1e3, 4))
        self.lineedit_speed.textEdited.connect(self.completeChanged)

        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setText(str(config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidatorNoZero(0, 1e3, 4))
        self.lineedit_scantime.textEdited.connect(self.completeChanged)

        # Krisskross params
        self.lineedit_warmup = QtWidgets.QLineEdit()
        self.lineedit_warmup.setText(str(config.warmup))
        self.lineedit_warmup.setValidator(DecimalValidator(0, 1e2, 2))
        self.lineedit_warmup.textEdited.connect(self.completeChanged)

        self.spinbox_offsets = QtWidgets.QSpinBox()
        self.spinbox_offsets.setRange(2, 10)
        self.spinbox_offsets.setValue(config._subpixel_size)
        self.spinbox_offsets.setToolTip(
            "The number of subpixels per pixel in each dimension."
        )

        # Form layout for line edits
        config_layout = QtWidgets.QFormLayout()
        config_layout.addRow("Spotsize (μm):", self.lineedit_spotsize)
        config_layout.addRow("Speed (μm):", self.lineedit_speed)
        config_layout.addRow("Scantime (s):", self.lineedit_scantime)

        config_gbox = QtWidgets.QGroupBox("Laser Configuration", self)
        config_gbox.setLayout(config_layout)

        params_layout = QtWidgets.QFormLayout()
        params_layout.addRow("Warmup (s):", self.lineedit_warmup)
        params_layout.addRow("Subpixel width:", self.spinbox_offsets)

        params_gbox = QtWidgets.QGroupBox("SRRLaser Parameters", self)
        params_gbox.setLayout(params_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(config_gbox)
        layout.addWidget(params_gbox)
        self.setLayout(layout)

        self.registerField("spotsize", self.lineedit_spotsize)
        self.registerField("speed", self.lineedit_speed)
        self.registerField("scantime", self.lineedit_scantime)
        self.registerField("warmup", self.lineedit_warmup)
        self.registerField("subpixel_width", self.spinbox_offsets)

    def isComplete(self) -> bool:
        return all(
            [
                self.lineedit_spotsize.hasAcceptableInput(),
                self.lineedit_speed.hasAcceptableInput(),
                self.lineedit_scantime.hasAcceptableInput(),
                self.lineedit_warmup.hasAcceptableInput(),
            ]
        )


class PathsSelectionWidget(QtWidgets.QWidget):
    pathsChanged = QtCore.Signal()
    firstPathChanged = QtCore.Signal()

    def __init__(
        self,
        files: List[str],
        filetype: str,
        exts: List[str],
        parent: QtWidgets.QWidget = None,
        uses_directories: bool = False,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.setTextElideMode(QtCore.Qt.ElideLeft)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.list.model().rowsInserted.connect(self.pathsChanged)
        self.list.model().rowsRemoved.connect(self.pathsChanged)
        self.list.model().rowsMoved.connect(self.pathsChanged)

        self.action_remove = qAction(
            "list-remove", "Remove", "Remove the selected path.", self.removeSelected
        )

        self.button_path = QtWidgets.QPushButton("Add Files")
        self.button_dir = QtWidgets.QPushButton("Add All Files...")
        self.button_remove = qToolButton(action=self.action_remove)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button_path, 0, QtCore.Qt.AlignLeft)
        button_layout.addWidget(self.button_dir, 0, QtCore.Qt.AlignLeft)
        button_layout.addWidget(self.button_remove, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    @QtCore.Property("QStringList")
    def paths(self) -> List[str]:
        return [self.list.item(i).text() for i in range(0, self.list.count())]

    def addPaths(self, paths: List[str]) -> None:
        for path in paths:
            self.list.addItem(path)

    def addPathsFromDirectory(self, directory: str) -> None:
        files = os.listdir(directory)
        files.sort()
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in self.exts:
                self.list.addItem(os.path.join(directory, f))

    def buttonPathsPressed(self) -> QtWidgets.QFileDialog:
        item = self.list.currentItem()
        dirname = os.path.dirname(item.text()) if item is not None else ""
        dlg = QtWidgets.QFileDialog(self, "Select Files", dirname)
        if self.uses_directories:
            dlg.setAcceptMode(QtWidgets.QFileDialog.Directory)
            dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        else:
            dlg.setNameFilters([self.nameFilter(), "All Files(*)"])
            dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.fileSelected.connect(self.addPaths)
        dlg.open()
        return dlg

    def buttonDirectoryPressed(self) -> QtWidgets.QFileDialog:
        item = self.list.currentItem()
        dirname = os.path.dirname(item.text()) if item is not None else ""
        dlg = QtWidgets.QFileDialog(self, "Select Directory", dirname)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.addPathsFromDirectory)
        dlg.open()
        return dlg

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
                    name, ext = os.path.splitext(path)
                    if ext.lower() in self.exts:
                        self.list.addItem(path)
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if (
            event.key() == QtCore.Qt.Key_Delete
            or event.key() == QtCore.Qt.Key_Backspace
        ):
            self.removeSelected()
        super().keyPressEvent(event)

    def nameFilter(self) -> str:
        return f"{self.filetype}({' '.join(['*' + ext for ext in self.exts])})"

    def removeSelected(self) -> None:
        items = self.list.selectedItems()
        for item in items:
            self.list.takeItem(item)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    w = SRRPathsWidget(["a", "b"], "a", [".x"])
    w.show()
    app.exec_()
