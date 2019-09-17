import os.path

from PySide2 import QtCore, QtGui, QtWidgets

from laserlib import io
from laserlib.laser import Laser

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import LaserCanvas
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

    def imagesize(self) -> Tuple[int, int]:
        return tuple(int(le.text()) for le in self.linedits)


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

    def spacing(self) -> Tuple[float, float, float]:
        return tuple(float(le.text()) for le in self.linedits)


class ExportOptions(QtWidgets.QStackedWidget):
    inputChanged = QtCore.Signal()

    def __init__(
        self,
        view_limits: Tuple[float, float, float, float],
        spacing: Tuple[float, float, float],
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        self.currentChanged.connect(self.inputChanged)

        self.npz = self.addWidget(OptionsBox("Numpy Archives", ".npz"))
        self.csv = self.addWidget(OptionsBox("CSV Documents", ".csv"))
        self.png = self.addWidget(
            PngOptionsBox(self.bestImageSize(view_limits, (1280, 800)))
        )
        self.vti = self.addWidget(VtiOptionsBox(spacing))

        for i in range(0, self.count()):
            self.widget(i).inputChanged.connect(self.inputChanged)

    def sizeHint(self) -> QtCore.QSize:
        sizes = [self.widget(i).sizeHint() for i in range(0, self.count())]
        return QtCore.QSize(
            max(s.width() for s in sizes), max(s.height() for s in sizes)
        )

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
    def __init__(
        self,
        laser: Laser,
        isotope: str,
        viewlimits: Tuple[float, float, float, float],
        viewoptions: ViewOptions,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.laser = laser
        self.isotope = isotope
        self.viewlimits = viewlimits
        self.viewoptions = viewoptions

        path = os.path.join(
            os.path.dirname(self.laser.filepath), self.laser.name + ".npz"
        )
        # path = laser.filepath
        directory = os.path.dirname(path)
        filename = os.path.basename(path)

        self.lineedit_directory = QtWidgets.QLineEdit(directory)
        self.lineedit_directory.setMinimumWidth(300)
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

        self.lineedit_preview = QtWidgets.QLineEdit()
        self.lineedit_preview.setEnabled(False)

        spacing = (
            laser.config.get_pixel_width(),
            laser.config.get_pixel_height(),
            laser.config.spotsize / 2.0,
        )
        self.options = ExportOptions(viewlimits, spacing)
        self.options.inputChanged.connect(self.validate)

        self.combo_type = QtWidgets.QComboBox()
        for i in range(0, self.options.count()):
            item = f"{self.options.widget(i).filetype} ({self.options.widget(i).ext})"
            self.combo_type.addItem(item)
        self.combo_type.setCurrentIndex(-1)
        self.combo_type.currentIndexChanged.connect(self.typeChanged)

        self.check_calibrate = QtWidgets.QCheckBox("Calibrate data.")
        self.check_calibrate.setChecked(True)
        self.check_calibrate.setToolTip("Calibrate the data before exporting.")

        self.check_export_all = QtWidgets.QCheckBox("Export all isotopes.")
        self.check_export_all.setToolTip(
            "Export all isotopes for the current image.\n"
            "The filename will be appended with the isotopes name."
        )
        self.check_export_all.stateChanged.connect(self.updatePreview)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout_directory = QtWidgets.QHBoxLayout()
        layout_directory.addWidget(self.lineedit_directory)
        layout_directory.addWidget(self.button_directory)

        layout = QtWidgets.QVBoxLayout()
        self.layout_form = QtWidgets.QFormLayout()
        self.layout_form.addRow("Directory:", layout_directory)
        self.layout_form.addRow("Filename:", self.lineedit_filename)
        self.layout_form.addRow("Preview:", self.lineedit_preview)
        self.layout_form.addRow("Type:", self.combo_type)

        layout.addLayout(self.layout_form)
        layout.addWidget(self.options)
        layout.addWidget(self.check_calibrate)
        layout.addWidget(self.check_export_all)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        # Init with correct filename and type
        self.filenameChanged(filename)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 200)

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

    def isCalibrate(self) -> bool:
        return self.check_calibrate.isChecked() and self.check_calibrate.isEnabled()

    def isExportAll(self) -> bool:
        return self.check_export_all.isChecked() and self.check_export_all.isEnabled()

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
        self.options.setVisible(
            self.options.currentIndex() in [self.options.png, self.options.vti]
        )
        self.adjustSize()
        # Enable or disable checks
        self.check_calibrate.setEnabled(self.options.allowCalibrate())
        self.check_export_all.setEnabled(self.options.allowExportAll())
        # Update name of file
        base, ext = os.path.splitext(self.lineedit_filename.text())
        if ext != "":
            ext = self.options.currentWidget().ext
        self.lineedit_filename.setText(base + ext)

    def selectDirectory(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(self, "Select Directory", "")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
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

    def generatePaths(self, laser: Laser) -> List[Tuple[str, str]]:
        if self.isExportAll():
            paths = [(self.getPathForIsotope(i), i) for i in laser.isotopes]
        else:
            paths = [(self.getPath(), self.isotope)]

        return [(p, i) for p, i in paths if p != ""]

    def export(
        self,
        paths: List[str],
        laser: Laser,
        viewlimits: Tuple[float, float, float, float],
    ) -> bool:
        index = self.options.currentIndex()
        try:
            if index == self.options.csv:
                kwargs = {"calibrate": self.isCalibrate(), "flat": True}
                for path, isotope in paths:
                    if isotope in laser.isotopes:
                        data = laser.get(isotope, **kwargs)
                        io.csv.save(path, data)

            elif index == self.options.png:
                x, y = self.options.widget(index).imagesize()
                canvas = LaserCanvas(self.viewoptions, self)
                dpi = canvas.figure.get_dpi()
                canvas.figure.set_size_inches(x / dpi, y / dpi)
                for path, isotope in paths:
                    if isotope in laser.isotopes:
                        canvas.drawLaser(laser, isotope)
                        canvas.view_limits = viewlimits
                        canvas.figure.savefig(path, transparent=True, facecolor=None)

                canvas.close()

            elif index == self.options.vti:
                spacing = self.options.widget(index).spacing()
                data = laser.get_structured(calibrate=self.isCalibrate())
                io.vtk.save(paths[0][0], data, spacing)

            else:  # npz
                io.npz.save(paths[0][0], [laser])
        except io.error.LaserLibException as e:
            QtWidgets.QMessageBox.critical(self, "Unable to Export!", str(e))
            return False

        return True

    def accept(self) -> None:
        paths = self.generatePaths(self.laser)
        prompt = OverwriteFilePrompt()
        paths = [p for p in paths if prompt.promptOverwrite(p[0])]
        if len(paths) == 0:
            return
        if self.export(paths, self.laser, self.viewlimits):
            super().accept()


class ExportAllDialog(ExportDialog):
    def __init__(
        self,
        lasers: List[Laser],
        isotopes: List[str],
        viewoptions: ViewOptions,
        parent: QtWidgets.QWidget = None,
    ):
        self.lasers = lasers
        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.addItems(isotopes)
        self.lineedit_prefix = QtWidgets.QLineEdit("")
        self.lineedit_prefix.textChanged.connect(self.updatePreview)
        super().__init__(
            Laser(name="<NAME>", filepath=lasers[0].filepath),
            "",
            (0, 1, 0, 1),
            viewoptions,
            parent,
        )
        # Adjust widgets for all
        label = self.layout_form.labelForField(self.lineedit_filename)
        label.setText("Prefix:")
        self.layout_form.replaceWidget(self.lineedit_filename, self.lineedit_prefix)

        layout_isotopes = QtWidgets.QHBoxLayout()
        layout_isotopes.addWidget(QtWidgets.QLabel("Isotope:"))
        layout_isotopes.addWidget(self.combo_isotopes)
        self.layout().insertLayout(2, layout_isotopes)

        self.check_export_all.stateChanged.connect(self.showIsotopes)
        self.showIsotopes()

    def showIsotopes(self) -> None:
        self.combo_isotopes.setEnabled(
            self.options.allowExportAll() and not self.isExportAll()
        )

    def typeChanged(self, index: int) -> None:
        super().typeChanged(index)
        self.showIsotopes()

    def updatePreview(self) -> None:
        base, ext = os.path.splitext(self.lineedit_filename.text())
        prefix = self.lineedit_prefix.text()
        if prefix != "":
            prefix += "_"
        if self.isExportAll():
            base += "_<ISOTOPE>"
        self.lineedit_preview.setText(prefix + base + ext)

    def getPath(self, name: str) -> str:
        base, ext = os.path.splitext(self.lineedit_filename.text())
        prefix = self.lineedit_prefix.text()
        if prefix != "":
            prefix += "_"
        return os.path.join(
            self.lineedit_directory.text(), f"{prefix}{base}_{name}{ext}"
        )

    def getPathForIsotope(self, isotope: str, name: str) -> str:
        base, ext = os.path.splitext(self.getPath(name))
        isotope = isotope.replace(os.path.sep, "_")
        return f"{base}_{isotope}{ext}"

    def generatePaths(self, laser: Laser) -> List[Tuple[str, str]]:
        if self.isExportAll():
            paths = [(self.getPathForIsotope(i, laser.name), i) for i in laser.isotopes]
        else:
            paths = [(self.getPath(laser.name), self.isotope)]

        return [(p, i) for p, i in paths if p != ""]

    def accept(self) -> None:
        allpaths = []
        prompt = OverwriteFilePrompt()
        self.isotope = self.combo_isotopes.currentText()
        for laser in self.lasers:
            paths = self.generatePaths(laser)
            allpaths.append([p for p in paths if prompt.promptOverwrite(p[0])])

        if any(len(p) == 0 for p in allpaths):
            return

        for paths, laser, viewlimits in zip(allpaths, self.lasers, self.viewlimits):
            if not self.export(paths, laser, viewlimits):
                return

        QtWidgets.QDialog.accept()
