import re
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.lib.recfunctions as rfn
from pewlib import io
from pewlib.process.calc import search_sorted_closest
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.events import DragDropRedirectFilter
from pewpew.widgets.ext import MultipleDirDialog
from pewpew.widgets.periodictable import PeriodicTableSelector, isotope_data


class _OptionsBase(QtWidgets.QGroupBox):  # pragma: no cover
    optionsChanged = QtCore.Signal()

    def __init__(
        self,
        filetype: str,
        filemode: str,
        exts: list[str],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Import Options", parent)
        self.filetype = filetype
        self.filemode = filemode
        self.exts = exts

    def fieldArgs(self) -> list[tuple[str, QtWidgets.QWidget, str, str]]:
        return []

    def isComplete(self) -> bool:
        return True

    def setEnabled(self, enabled: bool) -> None:
        pass

    def updateForPath(self, path: Path) -> None:
        pass


class AgilentOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Agilent Batch", "Directory", [".b"], parent)

        self.current_path = Path()
        self.actual_datafiles = 0
        self.expected_datafiles = -1

        self.combo_dfile_method = QtWidgets.QComboBox()
        self.combo_dfile_method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.combo_dfile_method.activated.connect(self.countDatafiles)
        self.combo_dfile_method.activated.connect(self.optionsChanged)

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
                self.current_path.joinpath(io.agilent.acq_method_xml_path),
            )
        elif method == "Batch Log CSV":
            datafiles = io.agilent.batch_csv_read_datafiles(
                self.current_path,
                self.current_path.joinpath(io.agilent.batch_csv_path),
            )
        elif method == "Batch Log XML":
            datafiles = io.agilent.batch_xml_read_datafiles(
                self.current_path,
                self.current_path.joinpath(io.agilent.batch_xml_path),
            )
        else:
            raise ValueError("Unknown data file collection method.")  # pragma: no cover

        csvs = [df.joinpath(df.with_suffix(".csv").name) for df in datafiles]

        self.actual_datafiles = sum([csv.exists() for csv in csvs])
        self.expected_datafiles = len(datafiles)

        if self.expected_datafiles == 0:
            self.lineedit_dfile.clear()
        else:
            self.lineedit_dfile.setText(
                f"{self.actual_datafiles} ({self.expected_datafiles} expected)"
            )

    def fieldArgs(self) -> list[tuple[str, QtWidgets.QWidget, str, str]]:
        return [
            (
                "method",
                self.combo_dfile_method,
                "currentText",
                "currentTextChanged",
            ),
            ("useAcqNames", self.check_name_acq_xml, "checked", "toggled"),
        ]

    def isComplete(self) -> bool:
        return self.actual_datafiles > 0

    def setEnabled(self, enabled: bool) -> None:
        self.combo_dfile_method.setEnabled(enabled)
        self.check_name_acq_xml.setEnabled(enabled)

        if not enabled:
            self.lineedit_dfile.clear()

    def updateForPath(self, path: Path) -> None:
        self.current_path = path

        self.combo_dfile_method.clear()

        self.combo_dfile_method.addItem("Alphabetical Order")
        if path.joinpath(io.agilent.acq_method_xml_path).exists():
            self.combo_dfile_method.addItem("Acquistion Method")
            self.check_name_acq_xml.setEnabled(True)
        else:
            self.check_name_acq_xml.setEnabled(False)
        if path.joinpath(io.agilent.batch_csv_path).exists():
            self.combo_dfile_method.addItem("Batch Log CSV")
        if path.joinpath(io.agilent.batch_xml_path).exists():
            self.combo_dfile_method.addItem("Batch Log XML")

        self.combo_dfile_method.setCurrentIndex(self.combo_dfile_method.count() - 1)

        self.countDatafiles()


class CsvLinesOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("CSV Lines", "Directory", [""], parent)
        self.csvs: list[Path] = []
        self.lines = 0

        self.option = io.csv.GenericOption()

        # Kwargs
        self.combo_delimiter = QtWidgets.QComboBox()
        self.combo_delimiter.addItems([",", ";", "Tab", "Space"])
        self.spinbox_header = QtWidgets.QSpinBox()
        self.spinbox_header.valueChanged.connect(self.updateHeaderPreview)
        self.spinbox_footer = QtWidgets.QSpinBox()

        self.lineedit_header_preview = QtWidgets.QLineEdit()
        self.lineedit_header_preview.setReadOnly(True)

        self.lineedit_regex = QtWidgets.QLineEdit(".*\\.csv")
        self.lineedit_regex.editingFinished.connect(self.regexChanged)
        self.lineedit_regex.editingFinished.connect(self.optionsChanged)
        self.combo_sortkey = QtWidgets.QComboBox()
        self.combo_sortkey.addItems(
            ["Alphabetical", "Numerical", "Timestamp"]  # , "Regex Match"]
        )
        self.combo_sortkey.currentIndexChanged.connect(self.sortingChanged)
        self.lineedit_sortkey = QtWidgets.QLineEdit()
        self.lineedit_sortkey.textChanged.connect(self.optionsChanged)

        self.lineedit_nlines = QtWidgets.QLineEdit("0")
        self.lineedit_nlines.setReadOnly(True)

        self.check_remove_empty_cols = QtWidgets.QCheckBox("Remove empty columns.")
        self.check_remove_empty_cols.setChecked(True)
        self.check_remove_empty_rows = QtWidgets.QCheckBox("Remove empty rows.")
        self.check_remove_empty_rows.setChecked(True)

        layout_header = QtWidgets.QHBoxLayout()
        layout_header.addWidget(self.spinbox_header, 0)
        layout_header.addWidget(self.lineedit_header_preview)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Delimiter:", self.combo_delimiter)
        layout.addRow("Header Rows:", layout_header)
        layout.addRow("Footer Rows:", self.spinbox_footer)
        layout.addRow("File Regex:", self.lineedit_regex)
        layout.addRow("Matching files:", self.lineedit_nlines)
        layout.addRow("Sorting:", self.combo_sortkey)
        layout.addRow("Sort key:", self.lineedit_sortkey)
        layout.addRow(self.check_remove_empty_cols)
        layout.addRow(self.check_remove_empty_rows)
        self.setLayout(layout)

    def fieldArgs(self) -> list[tuple[str, QtWidgets.QWidget, str, str]]:
        return [
            ("skipHeader", self.spinbox_header, "value", "valueChanged"),
            ("skipFooter", self.spinbox_footer, "value", "valueChanged"),
            ("delimiter", self.combo_delimiter, "currentText", "currentTextChanged"),
            ("regex", self.lineedit_regex, "text", "textChanged"),
            ("sorting", self.combo_sortkey, "currentText", "currentTextChanged"),
            ("sortKey", self.combo_sortkey, "currentText", "currentTextChanged"),
            ("removeEmptyCols", self.check_remove_empty_cols, "checked", "toggled"),
            ("removeEmptyRows", self.check_remove_empty_rows, "checked", "toggled"),
        ]

    def isComplete(self) -> bool:
        if (
            self.combo_sortkey.currentText() == "Timestamp"
            and "%" not in self.lineedit_sortkey.text()
        ):
            return False
        return self.lines > 1

    def regexChanged(self) -> None:
        regex = self.lineedit_regex.text()
        if regex == "":
            regex = ".*\\.csv"
        self.lines = sum([re.match(regex, p.name) is not None for p in self.csvs])
        self.lineedit_nlines.setText(f"{self.lines}")

    def sortingChanged(self) -> None:
        sorting = self.combo_sortkey.currentText()
        if sorting == "Timestamp":
            self.lineedit_sortkey.setEnabled(True)
            self.lineedit_sortkey.setText("%Y-%m-%d %H:%M:%S")
        else:
            self.lineedit_sortkey.setEnabled(False)
            self.lineedit_sortkey.setText("")

    def updateHeaderPreview(self, header: int) -> None:
        try:
            with open(self.csvs[0], "r") as fp:
                line = None
                for _ in range(header + 1):
                    line = fp.readline()
            self.lineedit_header_preview.setText(line)
        except (IndexError, ValueError):
            self.lineedit_header_preview.setText("")

    def updateForPath(self, path: Path) -> None:
        option = io.csv.option_for_path(path)
        if isinstance(option, io.csv.NuOption):
            self.lineedit_regex.setText(option.regex.pattern)
            self.combo_sortkey.setCurrentText("Numerical")
            self.combo_delimiter.setCurrentText(",")
            self.spinbox_header.setValue(11)
        elif isinstance(option, io.csv.TofwerkOption):
            self.lineedit_regex.setText(option.regex.pattern)
            self.combo_sortkey.setCurrentText("Timestamp")
            self.lineedit_sortkey.setText("%Y.%m.%d-%Hh%Mm%Ss")
            self.spinbox_header.setValue(0)
        elif isinstance(option, io.csv.ThermoLDROption):
            self.lineedit_regex.setText(option.regex.pattern)
            self.combo_delimiter.setCurrentText(",")
            self.spinbox_header.setValue(13)
            self.combo_sortkey.setCurrentText("Numerical")

        self.csvs = list(path.glob("*.csv"))
        self.regexChanged()
        self.updateHeaderPreview(self.spinbox_header.value())


class NuOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Nu Vitesse", "Directory", [""], parent)

        self.table = PeriodicTableSelector()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def fieldArgs(self) -> list[tuple[str, QtWidgets.QWidget, str, str]]:
        return [
            (
                "selectedIsotopes",
                self.table,
                "selectedIsotopes",
                "isotopesChanged",
            )
        ]

    def isComplete(self) -> bool:
        selected = self.table.selectedIsotopes()
        return selected is not None and selected.size > 0

    def updateForPath(self, path: Path):
        dir = next(
            d for d in path.iterdir() if d.is_dir() and d.joinpath("run.info").exists()
        )
        signals, masses, times, pulses, info = io.nu.read_laser_directory(
            dir, autoblank=False, max_integs=1
        )
        idx = search_sorted_closest(masses, isotope_data["mass"])
        selected = isotope_data[np.abs(masses[idx] - masses["mass"]) < 0.05]
        self.table.setEnabledIsotopes(selected)


class NumpyOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Numpy Archive", "File", [".npz"], parent)
        self.check_calibration = QtWidgets.QCheckBox("Import calibration.")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_calibration)
        self.setLayout(layout)

    def fieldArgs(self) -> list[tuple[str, QtWidgets.QWidget, str, str]]:
        return [("useCalibration", self.check_calibration, "checked", "toggled")]


class PerkinElmerOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Perkin-Elmer 'XL'", "Directory", [""], parent)
        self.datafiles = 0

        self.lineedit_dfile = QtWidgets.QLineEdit("0")
        self.lineedit_dfile.setReadOnly(True)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Datafiles found:", self.lineedit_dfile)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        return self.datafiles > 0

    def updateForPath(self, path: Path) -> None:
        self.datafiles = len(list(path.glob("*.xl")))
        self.lineedit_dfile.setText(str(self.datafiles))


class TextOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Text Image", "File", [".csv", ".text", ".txt"], parent)

        self.lineedit_name = QtWidgets.QLineEdit("_Element_")
        layout = QtWidgets.QFormLayout()
        layout.addRow("Element Name:", self.lineedit_name)

        self.setLayout(layout)

    def fieldArgs(self) -> list[tuple[str, QtWidgets.QWidget, str, str]]:
        return [("name", self.lineedit_name, "text", "textChanged")]

    def isComplete(self) -> bool:
        return self.lineedit_name.text() != ""


class ThermoOptions(_OptionsBase):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Thermo iCap Data", "File", [".csv"], parent)

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

    def fieldArgs(self) -> list[tuple[str, QtWidgets.QWidget, str, str]]:
        return [
            ("delimiter", self.combo_delimiter, "currentText", "currentTextChanged"),
            ("decimal", self.combo_decimal, "currentText", "currentTextChanged"),
            ("sampleColumns", self.radio_columns, "checked", "toggled"),
            ("sampleRows", self.radio_rows, "checked", "toggled"),
            ("useAnalog", self.check_use_analog, "checked", "toggled"),
        ]

    def isComplete(self) -> bool:
        return self.radio_rows.isChecked() or self.radio_columns.isChecked()

    def preprocessFile(self, path: Path) -> tuple[str, str, bool]:
        method = "unknown"
        has_analog = False
        with path.open("r", encoding="utf-8-sig") as fp:
            lines = [next(fp) for _ in range(3)]
            delimiter = lines[0][0]
            if "MainRuns" in lines[0]:  # pragma: no cover
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

    def updateForPath(self, path: Path) -> None:
        delimiter, method, has_analog = self.preprocessFile(path)
        self.combo_delimiter.setCurrentText(delimiter)
        if method == "rows":  # pragma: no cover
            self.radio_rows.setChecked(True)
        elif method == "columns":
            self.radio_columns.setChecked(True)
        else:
            self.radio_rows.setChecked(False)
            self.radio_rows.setAutoExclusive(False)
            self.radio_columns.setChecked(False)
            self.radio_rows.setAutoExclusive(True)

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
        exts: list[str],
        mode: str = "File",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        if mode not in ["File", "Directory"]:  # pragma: no cover
            raise ValueError("Mode must be one of 'File', 'Directory'.")
        self.mode = mode
        self.filetype = filetype
        self.exts = exts

        self.setAcceptDrops(True)

    @QtCore.Property("QString")  # type: ignore
    def _path(self) -> str:  # pragma: no cover
        raise NotImplementedError

    @property
    def path(self) -> Path:
        return Path(self._path)  # type: ignore

    @QtCore.Property("QStringList")  # type: ignore
    def _paths(self) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    @property
    def paths(self) -> list[Path]:
        return [Path(p) for p in self._paths]  # type: ignore

    def addPath(self, path: str | Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def addPaths(self, paths: list[str] | list[Path]) -> None:  # pragma: no cover
        raise NotImplementedError

    def selectPath(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(
            self, f"Select {self.mode}", str(self.currentDir().resolve())
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        if self.mode == "File":
            dlg.setNameFilters([self.nameFilter(), "All Files(*)"])
            dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        else:
            dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
            dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.addPath)
        dlg.open()
        return dlg

    def selectMultiplePaths(self) -> QtWidgets.QFileDialog:
        if self.mode == "File":
            dlg = QtWidgets.QFileDialog(
                self, "Select Files", str(self.currentDir().resolve())
            )
            dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
            dlg.setNameFilters([self.nameFilter(), "All Files(*)"])
            dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        else:
            dlg = MultipleDirDialog(
                self, "Select Directories", str(self.currentDir().resolve())
            )
        dlg.filesSelected.connect(self.addPaths)
        dlg.open()
        return dlg

    def currentDir(self) -> Path:
        return self.path

    def isComplete(self) -> bool:
        if len(self.paths) == 0:
            return False
        return all([self.validPath(path) for path in self.paths])

    def nameFilter(self) -> str:
        return f"{self.filetype}({' '.join(['*' + ext for ext in self.exts])})"

    def validPath(self, path: Path) -> bool:
        if not path.exists():
            return False
        if path.suffix.lower() not in [ext.lower() for ext in self.exts]:
            return False
        if self.mode == "File":
            return path.is_file()
        else:
            return path.is_dir()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:  # pragma: no cover
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if self.validPath(path):
                        self.addPath(path)
            event.acceptProposedAction()
        else:  # pragma: no cover
            super().dropEvent(event)


class PathSelectWidget(_PathSelectBase):
    def __init__(
        self,
        path: Path,
        filetype: str,
        exts: list[str],
        mode: str = "File",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(filetype, exts, mode, parent)

        # dumb check to see if 'empty' path is used
        if path == Path():
            text = ""
        else:
            text = str(path.resolve())

        self.lineedit_path = QtWidgets.QLineEdit(text)
        self.lineedit_path.setPlaceholderText(f"Path to {self.mode}...")
        self.lineedit_path.textChanged.connect(self.pathChanged)
        self.lineedit_path.installEventFilter(DragDropRedirectFilter(self))

        self.button_path = QtWidgets.QPushButton(f"Open {self.mode}")
        self.button_path.pressed.connect(self.selectPath)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.lineedit_path, 1)
        layout.addWidget(self.button_path, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        self.setLayout(layout)

    @QtCore.Property("QString")  # type: ignore
    def _path(self) -> str:
        return self.lineedit_path.text()

    @QtCore.Property("QStringList")  # type: ignore
    def _paths(self) -> list[str]:
        return [self.lineedit_path.text()]

    def addPath(self, path: str | Path) -> None:
        if isinstance(path, Path):
            path = str(path.resolve())
        self.lineedit_path.setText(path)


class MultiplePathSelectWidget(_PathSelectBase):
    def __init__(
        self,
        paths: list[Path],
        filetype: str,
        exts: list[str],
        mode: str = "File",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(filetype, exts, mode, parent)

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.list.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)

        self.list.model().rowsInserted.connect(self.pathChanged)
        self.list.model().rowsRemoved.connect(self.pathChanged)
        self.list.model().rowsMoved.connect(self.pathChanged)

        self.addPaths(paths)

        text = "Files" if self.mode == "File" else "Directories"
        self.button_path = QtWidgets.QPushButton(f"Open {text}")
        self.button_path.pressed.connect(self.selectMultiplePaths)
        self.button_dir = QtWidgets.QPushButton("Open All...")
        self.button_dir.pressed.connect(self.selectAllInDirectory)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.button_path, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        button_layout.addWidget(self.button_dir, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.list)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.setLayout(layout)

    @QtCore.Property("QString")  # type: ignore
    def _path(self) -> str:
        first = self.list.item(0)
        return first.text() if first is not None else ""

    @QtCore.Property("QStringList")  # type: ignore
    def _paths(self) -> list[str]:
        return [self.list.item(i).text() for i in range(0, self.list.count())]

    def addPath(self, path: str | Path) -> None:
        if isinstance(path, Path):
            path = str(path.resolve())
        self.list.addItem(path)

    def addPaths(self, paths: list[str] | list[Path]) -> None:
        for path in paths:
            if isinstance(path, Path):
                path = str(path.resolve())
            self.list.addItem(path)

    def addPathsInDirectory(self, directory: Path) -> None:
        files = []
        for it in directory.iterdir():
            if self.validPath(it):
                files.append(str(it.resolve()))
        self.list.addItems(sorted(files))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in [QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete]:
            items = self.list.selectedItems()
            for item in items:
                self.list.takeItem(self.list.row(item))

    def selectAllInDirectory(self) -> QtWidgets.QFileDialog:
        dlg = QtWidgets.QFileDialog(
            self, "Select Directory", str(self.currentDir().resolve())
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.addPathsInDirectory)
        dlg.open()
        return dlg


class PathAndOptionsPage(QtWidgets.QWizardPage):
    formats = {
        "agilent": (
            (
                "Agilent Batch",
                [".b"],
                "Directory",
                "Select the batch directory.",
            ),
            AgilentOptions,
        ),
        "csv": (
            (
                "CSV Lines",
                [""],
                "Directory",
                "Select the directory containing exported files.\n"
                "Files should have elemental data in columns.",
            ),
            CsvLinesOptions,
        ),
        "nu": (
            (
                "Nu Vitesse Images",
                [""],
                "Directory",
                "Select Nu images directories.\nThese have the format 'Image00X'.",
            ),
            NuOptions,
        ),
        "numpy": (
            ("Numpy Archive", [".npz"], "File", "Select Pew² export."),
            NumpyOptions,
        ),
        "perkinelmer": (
            (
                "Perkin-Elmer 'XL'",
                [""],
                "Directory",
                "Select export directory containing '.xl' files.",
            ),
            PerkinElmerOptions,
        ),
        "text": (
            (
                "Text Image",
                [".csv", ".text", ".txt"],
                "File",
                "Import a 2D delimited text file.",
            ),
            TextOptions,
        ),
        "thermo": (
            ("Thermo iCap Data", [".csv"], "File", "Select the Qtegra '.csv' export."),
            ThermoOptions,
        ),
    }

    dataChanged = QtCore.Signal()
    paramsChanged = QtCore.Signal()
    infoChanged = QtCore.Signal()

    def __init__(
        self,
        paths: list[Path],
        format: str,
        multiplepaths: bool = False,
        nextid: int | None = None,
        register_laser_fields: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        (ftype, exts, fmode, fdesc), otype = self.formats[format]
        self.setTitle(ftype + " Import")
        self.nextid = nextid

        self._laser_datas: list[np.ndarray] = []
        self._laser_params: list[dict] = []
        self._laser_infos: list[dict] = []

        if multiplepaths:
            self.path = MultiplePathSelectWidget(paths, ftype, exts, fmode)
        else:
            self.path = PathSelectWidget(paths[0], ftype, exts, fmode)
        self.path.pathChanged.connect(self.updateOptionsForPath)
        self.path.pathChanged.connect(self.completeChanged)

        self.options = otype()
        self.options.optionsChanged.connect(self.completeChanged)
        self.setAcceptDrops(True)
        self.installEventFilter(DragDropRedirectFilter(self.path))

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel(fdesc), 0)
        layout.addWidget(self.path, 0)
        layout.addWidget(self.options, 1)
        self.setLayout(layout)

        for name, widget, prop, signal in self.options.fieldArgs():
            self.registerField(format + "." + name, widget, prop, signal)

        self.registerField(format + ".path", self.path, "_path")
        self.registerField(format + ".paths", self.path, "_paths")

        if register_laser_fields:
            self.registerField("laserdata", self, "data_prop")
            self.registerField("laserparam", self, "param_prop")
            self.registerField("laserinfo", self, "info_prop")

    def cleanupPage(self) -> None:
        pass

    def initializePage(self) -> None:
        self.updateOptionsForPath()

    def isComplete(self) -> bool:
        return self.path.isComplete() and self.options.isComplete()

    def nextId(self) -> int:  # pragma: no cover
        if self.nextid is not None:
            return self.nextid
        return super().nextId()

    def updateOptionsForPath(self) -> None:
        if self.path.isComplete():
            self.options.setEnabled(True)
            self.options.updateForPath(self.path.path)
        else:
            self.options.setEnabled(False)

    def validatePage(self) -> bool:
        try:
            if self.field("agilent"):
                paths = [Path(p) for p in self.field("agilent.paths")]
                datas, params, infos = self.readMultiple(self.readAgilent, paths)
            elif self.field("csv"):
                paths = [Path(p) for p in self.field("csv.paths")]
                datas, params, infos = self.readMultiple(self.readCsv, paths)
            elif self.field("numpy"):
                paths = [Path(p) for p in self.field("numpy.paths")]
                datas, params, infos = self.readMultiple(self.readNumpy, paths)
            elif self.field("perkinelmer"):
                paths = [Path(p) for p in self.field("perkinelmer.paths")]
                datas, params, infos = self.readMultiple(self.readPerkinElmer, paths)
            elif self.field("text"):
                paths = [Path(p) for p in self.field("text.paths")]
                datas, params, infos = self.readMultiple(self.readText, paths)
            elif self.field("thermo"):
                paths = [Path(p) for p in self.field("thermo.paths")]
                datas, params, infos = self.readMultiple(self.readThermo, paths)
            else:
                raise ValueError

        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Import Error", str(e))
            return False

        self.setField("laserdata", datas)
        self.setField("laserparam", params)
        self.setField("laserinfo", infos)

        return True

    def readMultiple(
        self,
        func: Callable[[Path], tuple[np.ndarray, dict[str, Any], dict[str, str]]],
        paths: list[Path],
    ) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, str]]]:
        datas = []
        params = []
        infos = []
        for path in paths:
            data, param, info = func(path)
            datas.append(data)
            params.append(param)
            infos.append(info)
        for path, info in zip(paths, infos):
            info.update(
                {
                    "Name": path.stem,
                    "File Path": str(path.resolve()),
                    "Import Date": time.strftime(
                        "%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())
                    ),
                    "Import Path": str(path.resolve()),
                    "Import Version pewlib": version("pewlib"),
                    "Import Version pew2": version("pewpew"),
                }
            )
        return datas, params, infos

    def readAgilent(
        self, path: Path
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
        agilent_method = self.field("agilent.method")
        if agilent_method == "Alphabetical Order":  # pragma: no cover
            method = ["alphabetical"]  # Fallback to alphabetical
        elif agilent_method == "Acquistion Method":  # pragma: no cover
            method = ["acq_method_xml"]
        elif agilent_method == "Batch Log CSV":  # pragma: no cover
            method = ["batch_csv"]
        elif agilent_method == "Batch Log XML":
            method = ["batch_xml"]
        else:  # pragma: no cover
            raise ValueError("Unknown data file collection method.")

        data, params = io.agilent.load(
            path,
            collection_methods=method,
            use_acq_for_names=self.field("agilent.useAcqNames"),
            full=True,
        )
        info = io.agilent.load_info(path)
        return data, params, info

    def readCsv(self, path: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
        delimiter = self.field("csv.delimiter")
        if delimiter == "Tab":
            delimiter = "\t"
        elif delimiter == "Space":
            delimiter = " "

        option = io.csv.GenericOption(
            kw_genfromtxt={
                "delimiter": delimiter,
                "skip_header": self.field("csv.skipHeader"),
                "skip_footer": self.field("csv.skipFooter"),
            },
            regex=self.field("csv.regex"),
            drop_nan_columns=self.field("csv.removeEmptyCols"),
            drop_nan_rows=self.field("csv.removeEmptyRows"),
        )
        sorting = self.field("csv.sorting")
        if sorting == "Numerical":
            option.sortkey = lambda p: int("".join(filter(str.isdigit, p.stem)) or -1)
        elif sorting == "Timestamp":
            regex = re.sub(r"%\w", r"\d+", self.field("csv.sortKey"))
            retime = re.compile(regex)

            def sortTimestamp(path: Path) -> float:
                match = retime.search(path.name)
                if match is None:
                    return -1
                return time.mktime(
                    time.strptime(match.group(0), self.field("csv.sortKey"))
                )

            option.sortkey = sortTimestamp

        data, params = io.csv.load(path, option=option, full=True)
        return data, params, {}

    def readNu(self, path: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        isotopes = self.field("nu.selectedIsotopes")
        signals, masses, times = io.nu.read_laser_image(path)
        idx = search_sorted_closest(masses, isotopes["mass"])

        dtype = [(f"{iso['isotope']}{iso['symbol']}", float) for iso in isotopes]
        signals = rfn.unstructured_to_structured(signals[:, idx], dtype=dtype)

        params = {"masses": masses, "times": times}
        return signals, params, {}

    def readNumpy(
        self, path: Path
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
        laser = io.npz.load(path)
        param = {
            "scantime": laser.config.scantime,
            "speed": laser.config.speed,
            "spotsize": laser.config.spotsize,
        }
        return laser.data, param, laser.info

    def readPerkinElmer(
        self, path: Path
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
        data, params = io.perkinelmer.load(path, full=True)
        return data, params, {"Instrument Vendor": "PerkinElemer"}

    def readText(self, path: Path) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
        data = io.textimage.load(path, name=self.field("text.name"))
        return data, {}, {}

    def readThermo(
        self, path: Path
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, str]]:
        kwargs = dict(
            delimiter=self.field("thermo.delimiter"),
            comma_decimal=self.field("thermo.decimal") == ",",
        )
        use_analog = self.field("thermo.useAnalog")

        if self.field("thermo.sampleRows"):  # pragma: no cover
            data = io.thermo.icap_csv_rows_read_data(
                path, use_analog=use_analog, **kwargs
            )
            params = io.thermo.icap_csv_rows_read_params(path, **kwargs)
        else:
            data = io.thermo.icap_csv_columns_read_data(
                path, use_analog=use_analog, **kwargs
            )
            params = io.thermo.icap_csv_columns_read_params(path, **kwargs)
        return data, params, {"Instrument Vendor": "Thermo"}

    def getData(self) -> list[np.ndarray]:
        return self._laser_datas

    def setData(self, datas: list[np.ndarray]) -> None:
        self._laser_datas = datas
        self.dataChanged.emit()

    def getParams(self) -> list[dict]:
        return self._laser_params

    def setParams(self, params: list[dict]) -> None:
        self._laser_params = params
        self.paramsChanged.emit()

    def getInfo(self) -> list[dict]:
        return self._laser_infos

    def setInfo(self, infos: list[dict]) -> None:
        self._laser_infos = infos
        self.infoChanged.emit()

    data_prop = QtCore.Property("QVariant", getData, setData, notify=dataChanged)  # type: ignore
    param_prop = QtCore.Property("QVariant", getParams, setParams, notify=paramsChanged)  # type: ignore
    info_prop = QtCore.Property("QVariant", getInfo, setInfo, notify=infoChanged)  # type: ignore
