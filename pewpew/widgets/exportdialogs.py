import os.path

from PySide2 import QtCore, QtGui, QtWidgets

from laserlib import io
from laserlib.laser import Laser

from pewpew.widgets.prompts import OverwriteFilePrompt

from typing import List, Tuple


class OptionsBox(QtWidgets.QGroupBox):
    inputChanged = QtCore.Signal()

    def __init__(self, filetype: str, ext: str, parent: QtWidgets.QWidget = None):
        super().__init__("Format Options", parent)
        self.filetype = filetype
        self.ext = ext

    # Because you can't hook up signals with different no. of params
    def isComplete(self) -> bool:
        return True


class CsvOptionsBox(OptionsBox):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("CSV Documents", ".csv", parent)
        self.check_trim = QtWidgets.QCheckBox("Trim data to view.")
        self.check_trim.setChecked(True)
        self.check_trim.clicked.connect(self.inputChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_trim)
        self.setLayout(layout)

    # def getOptions(self) -> dict:
    #     return {
    #         "trim": self.check_trim.isChecked(),
    #         "calibrate": self.check_calibrate.isChecked(),
    #     }


class PngOptionsBox(OptionsBox):
    def __init__(self, imagesize: Tuple[int, int], parent: QtWidgets.QWidget = None):
        super().__init__("PNG Images", ".png", parent)
        self.linedits = [QtWidgets.QLineEdit(str(dim)) for dim in imagesize]
        for le in self.linedits:
            le.setValidator(QtGui.QIntValidator(0, 9999))
            le.textEdited.connect(self.inputChanged)

        layout_edits = QtWidgets.QHBoxLayout()
        layout_edits.addWidget(QtWidgets.QLabel("Size:"), 0)
        layout_edits.addWidget(self.linedits[0], 0)  # X
        layout_edits.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout_edits.addWidget(self.linedits[1], 0)  # Y
        layout_edits.addStretch(1)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_edits)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        return all(le.hasAcceptableInput() for le in self.linedits)


class VtiOptionsBox(OptionsBox):
    def __init__(
        self, spacing: Tuple[float, float, float], parent: QtWidgets.QWidget = None
    ):
        super().__init__("VTK Images", ".vti", parent)
        self.linedits = [QtWidgets.QLineEdit(str(dim)) for dim in spacing]
        for le in self.linedits:
            le.setValidator(QtGui.QDoubleValidator(-1e9, 1e9, 4))
            le.textEdited.connect(self.inputChanged)
        self.linedits[0].setEnabled(False)  # X
        self.linedits[1].setEnabled(False)  # Y

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Spacing:"), 0)
        layout.addWidget(self.linedits[0], 0)  # X
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.linedits[1], 0)  # Y
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.linedits[2], 0)  # Z
        layout.addStretch(1)

        self.setLayout(layout)

    def isComplete(self) -> bool:
        return all(le.hasAcceptableInput() for le in self.linedits)


class ExportOptions(QtWidgets.QStackedWidget):
    inputChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.currentChanged.connect(self.inputChanged)

        view_limits = (1, 2, 2, 4)
        spacing = (0, 10, 20)

        self.npz = self.addWidget(OptionsBox("Numpy Archives", ".npz"))
        self.csv = self.addWidget(CsvOptionsBox())
        self.png = self.addWidget(
            PngOptionsBox(self.bestImageSize(view_limits, (1280, 800)))
        )
        self.vti = self.addWidget(VtiOptionsBox(spacing))

        for i in range(0, self.count()):
            self.widget(i).inputChanged.connect(self.inputChanged)

    def bestImageSize(
        self, extents: Tuple[float, float, float, float], size: Tuple[int, int]
    ) -> Tuple[int, int]:
        x = extents[1] - extents[0]
        y = extents[3] - extents[2]
        return (
            (size[0], int(size[0] * x / y))
            if x > y
            else (int(size[1] * y / x), size[1])
        )

    def isComplete(self, current_only: bool = True) -> bool:
        indicies = [self.currentIndex()] if current_only else range(0, self.count())
        return all(self.widget(i).isComplete() for i in indicies)

    def allowCalibrate(self) -> bool:
        return self.currentIndex() != self.npz

    def allowExportAll(self) -> bool:
        return self.currentIndex() not in [self.npz, self.vti]


class ExportDialog(QtWidgets.QDialog):
    def __init__(self, path: str, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.path = path
        directory = os.path.dirname(path)
        filename = os.path.basename(path)

        self.lineedit_directory = QtWidgets.QLineEdit(directory)
        self.lineedit_directory.setClearButtonEnabled(True)
        self.lineedit_directory.textChanged.connect(self.validate)

        icon = QtGui.QIcon.fromTheme("document-open-folder")
        self.button_directory = QtWidgets.QPushButton(
            icon, "Open" if icon.isNull() else ""
        )
        self.button_directory.clicked.connect(self.selectDirectory)
        self.lineedit_filename = QtWidgets.QLineEdit(filename)
        self.lineedit_filename.textChanged.connect(self.filenameChanged)
        self.lineedit_filename.textChanged.connect(self.validate)

        self.options = ExportOptions()
        self.options.inputChanged.connect(self.validate)

        self.combo_type = QtWidgets.QComboBox()
        for i in range(0, self.options.count()):
            item = f"{self.options.widget(i).filetype} ({self.options.widget(i).ext})"
            self.combo_type.addItem(item)
        self.combo_type.currentIndexChanged.connect(self.typeChanged)

        self.lineedit_preview = QtWidgets.QLineEdit()
        self.lineedit_preview.setEnabled(False)

        self.check_calibrate = QtWidgets.QCheckBox("Calibrate data.")
        self.check_calibrate.setChecked(True)
        self.check_calibrate.setToolTip("Calibrate the data before exporting.")

        self.check_all_isotopes = QtWidgets.QCheckBox("Export all isotopes.")
        self.check_all_isotopes.setChecked(True)
        self.check_all_isotopes.setToolTip(
            "Export all isotopes for the current image.\n"
            "The filename will be appended with the isotopes name."
        )
        self.check_all_isotopes.clicked.connect(self.updatePreview)
        # self.check_all_isotopes.cha(self.updatePreview)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout_directory = QtWidgets.QHBoxLayout()
        layout_directory.addWidget(self.lineedit_directory)
        layout_directory.addWidget(self.button_directory)

        layout = QtWidgets.QVBoxLayout()
        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Directory:", layout_directory)
        layout_form.addRow("Filename:", self.lineedit_filename)
        layout_form.addRow("Preview:", self.lineedit_preview)
        layout_form.addRow("Type:", self.combo_type)

        layout.addLayout(layout_form)
        layout.addWidget(self.options)
        layout.addWidget(self.check_calibrate)
        layout.addWidget(self.check_all_isotopes)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        # Init with correct filename
        self.filenameChanged(filename)

    def isComplete(self) -> bool:
        if not os.path.exists(self.lineedit_directory.text()):
            return False
        if self.lineedit_filename.text() == "":
            return False
        if not self.options.isComplete():
            return False
        return True

    def validate(self) -> None:
        ok = self.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        ok.setEnabled(self.isComplete())

    def isExportAll(self) -> bool:
        return (
            self.check_all_isotopes.isChecked() and self.check_all_isotopes.isEnabled()
        )

    def updatePreview(self) -> None:
        base, ext = os.path.splitext(self.lineedit_filename.text())
        if self.isExportAll():
            base += "_<ISOTOPE>"
        self.lineedit_preview.setText(base + ext)

    def filenameChanged(self, filename: str) -> None:
        _, ext = os.path.splitext(filename.lower())
        if ext == ".csv":
            index = self.options.csv
        elif ext == ".npz":
            index = self.options.npz
        elif ext == ".png":
            index = self.options.png
        elif ext == ".vti":
            index = self.options.vti
        else:
            index = self.options.currentIndex()
        self.combo_type.setCurrentIndex(index)
        self.updatePreview()

    def typeChanged(self, index: int) -> None:
        self.options.setCurrentIndex(index)
        # Hide options when not needed
        self.options.setVisible(not self.options.currentIndex() == self.options.npz)
        # Enable or disable checks
        self.check_calibrate.setEnabled(self.options.allowCalibrate())
        self.check_all_isotopes.setEnabled(self.options.allowExportAll())
        # Update name of file
        base, ext = os.path.splitext(self.lineedit_filename.text())
        if ext != "":
            ext = self.options.currentWidget().ext
        self.lineedit_filename.setText(base + ext)

    def selectDirectory(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(self, "Select Directory", "")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, False)
        dlg.fileSelected.connect(self.lineedit_directory.setText)
        dlg.open()
        return dlg

    def getPath(self) -> str:
        return os.path.join(
            self.lineedit_directory.text(), self.lineedit_filename.text()
        )

    def getPathForIsotope(self, isotope: str) -> str:
        base, ext = os.path.splitext(self.getPath())
        isotope = isotope.replace(os.path.sep, "_")
        return f"{base}_{isotope}{ext}"

    def generatePaths(self) -> List[Tuple[str, str]]:
        if self.isExportAll():
            paths = [(self.getPathForIsotope(i), i) for i in self.laser.isotopes]
        else:
            paths = [(self.getPath(), self.isotope)]

        prompt = OverwriteFilePrompt()
        return [(p, i) for p, i in paths if p != "" and prompt.promptOverwrite(p)]

    def export(self, path: str) -> None:
        raise NotImplementedError


# class ExportDialog(QtWidgets.QDialog):
#     def __init__(
#         self,
#         path: str,
#         laser: Laser,
#         current_isotope: str,
#         options: ExportOptions,
#         defaults: dict = None,
#         parent: QtWidgets.QWidget = None,
#     ):
#         super().__init__(parent)
#         self.setWindowTitle("Export")
#         self.laser = laser
#         self.path = path
#         self.isotope = current_isotope

#         self.options = options
#         self.options.inputChanged.connect(self.optionsChanged)

#         self.check_isotopes = QtWidgets.QCheckBox("Export all isotopes.")
#         if len(laser.isotopes) < 2 or options == ".vti":
#             self.check_isotopes.setEnabled(False)
#         self.check_isotopes.stateChanged.connect(self.updatePreview)

#         self.lineedit_preview = QtWidgets.QLineEdit()
#         self.lineedit_preview.setEnabled(False)

#         self.button_box = QtWidgets.QDialogButtonBox(
#             QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
#         )
#         self.button_box.accepted.connect(self.accept)
#         self.button_box.rejected.connect(self.reject)

#         layout_preview = QtWidgets.QHBoxLayout()
#         layout_preview.addWidget(QtWidgets.QLabel("Preview:"))
#         layout_preview.addWidget(self.lineedit_preview)

#         layout_main = QtWidgets.QVBoxLayout()
#         layout_main.addWidget(self.options)
#         layout_main.addWidget(self.check_isotopes)
#         layout_main.addLayout(layout_preview)
#         layout_main.addWidget(self.button_box)
#         self.setLayout(layout_main)

#         self.updatePreview()

#     def optionsChanged(self) -> None:
#         ok = self.button_box.button(QtWidgets.QDialogButtonBox.Ok)
#         ok.setEnabled(self.options.isComplete())

#     def exportAllIsotopes(self) -> bool:
#         return self.check_isotopes.isChecked() and self.check_isotopes.isEnabled()

#     def updatePreview(self) -> None:
#         self.lineedit_preview.setText(
#             os.path.basename(
#                 self.generatePath(
#                     self.path, isotope=self.isotope if self.exportAllIsotopes() else ""
#                 )
#             )
#         )

#     def generatePath(self, path: str, isotope: str = "") -> str:
#         base, ext = os.path.splitext(path)
#         isotope = isotope.replace(os.path.sep, "_")
#         return f"{base}{'_' if isotope else ''}{isotope}{ext}"

#     def generatePaths(self, path: str) -> List[Tuple[str, str]]:
#         if self.exportAllIsotopes():
#             paths = [(self.generatePath(path, i), i) for i in self.laser.isotopes]
#         else:
#             paths = [(self.generatePath(path), self.isotope)]

#         prompt = OverwriteFilePrompt()
#         return [(p, i) for p, i in paths if p != "" and prompt.promptOverwrite(p)]

#     def export(self, path: str) -> None:
#         raise NotImplementedError
