import os.path

from PySide2 import QtCore, QtGui, QtWidgets

from laserlib import io
from laserlib.laser import Laser

from pewpew.widgets.canvases import LaserCanvas
from pewpew.widgets.prompts import OverwriteFilePrompt

from typing import List, Tuple


class ExportOptions(QtWidgets.QGroupBox):
    inputChanged = QtCore.Signal(str)

    def __init__(self, title: str, parent: QtWidgets.QWidget = None):
        super().__init__(title, parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

    def isComplete(self) -> bool:
        return True

    def getOptions(self) -> dict:
        return {}


class CsvExportOptions(ExportOptions):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Csv Options", parent=parent)
        self.check_trim = QtWidgets.QCheckBox("Trim data to view.")
        self.check_trim.setChecked(True)
        self.check_trim.clicked.connect(self.inputChanged.emit)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_trim)
        self.setLayout(layout)

    def getOptions(self) -> dict:
        return {"trim": self.check_trim.isChecked()}


class PngExportOptions(ExportOptions):
    def __init__(self, imagesize: Tuple[int, int], parent: QtWidgets.QWidget = None):
        super().__init__("Png Options", parent=parent)
        self.linedit_size_x = QtWidgets.QLineEdit(str(imagesize[0]))
        self.linedit_size_x.setValidator(QtGui.QIntValidator(0, 9999))
        self.linedit_size_x.textEdited.connect(self.inputChanged.emit)
        self.linedit_size_y = QtWidgets.QLineEdit(str(imagesize[1]))
        self.linedit_size_y.setValidator(QtGui.QIntValidator(0, 9999))
        self.linedit_size_y.textEdited.connect(self.inputChanged.emit)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Size:"), 0)
        layout.addWidget(self.linedit_size_x, 0)
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.linedit_size_y, 0)
        layout.addStretch(1)

        self.setLayout(layout)

    def isComplete(self) -> bool:
        return (
            self.linedit_size_x.hasAcceptableInput()
            and self.linedit_size_y.hasAcceptableInput()
        )

    def getOptions(self) -> dict:
        return {
            "imagesize": (
                int(self.linedit_size_x.text()),
                int(self.linedit_size_y.text()),
            )
        }


class VtiExportOptions(ExportOptions):
    def __init__(
        self, spacing: Tuple[float, float, float], parent: QtWidgets.QWidget = None
    ):
        super().__init__("Vti Options", parent=parent)
        self.linedit_size_x = QtWidgets.QLineEdit(str(spacing[0]))
        self.linedit_size_x.setEnabled(False)
        self.linedit_size_y = QtWidgets.QLineEdit(str(spacing[1]))
        self.linedit_size_y.setEnabled(False)
        self.linedit_size_z = QtWidgets.QLineEdit(str(spacing[2]))
        self.linedit_size_z.setValidator(QtGui.QDoubleValidator(-1e99, 1e99, 4))
        self.linedit_size_z.textEdited.connect(self.inputChanged.emit)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Spacing:"), 0)
        layout.addWidget(self.linedit_size_x, 0)
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.linedit_size_y, 0)
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.linedit_size_z, 0)
        layout.addStretch(1)

        self.setLayout(layout)

    def isComplete(self) -> bool:
        return self.linedit_size_z.hasAcceptableInput()

    def getOptions(self) -> dict:
        return {
            "spacing": (
                float(self.linedit_size_x.text()),
                float(self.linedit_size_y.text()),
                float(self.linedit_size_z.text()),
            )
        }


class ExportDialog(QtWidgets.QDialog):
    def __init__(
        self,
        laser: Laser,
        current_isotope: str,
        options: ExportOptions,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.isotope = current_isotope
        self.laser = laser

        self.options = options
        self.options.inputChanged.connect(self.optionsChanged)

        self.check_isotopes = QtWidgets.QCheckBox("Export all isotopes.")
        if len(laser.isotopes) < 2 or options == ".vti":
            self.check_isotopes.setEnabled(False)
        self.check_isotopes.stateChanged.connect(self.updatePreview)

        self.lineedit_preview = QtWidgets.QLineEdit()
        self.lineedit_preview.setEnabled(False)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout_preview = QtWidgets.QHBoxLayout()
        layout_preview.addWidget(QtWidgets.QLabel("Preview:"))
        layout_preview.addWidget(self.lineedit_preview)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addWidget(self.options)
        layout_main.addWidget(self.check_isotopes)
        layout_main.addLayout(layout_preview)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

        self.updatePreview()

    def optionsChanged(self) -> None:
        ok = self.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        ok.setEnabled(self.options.isComplete())

    def exportAllIsotopes(self) -> bool:
        return self.check_isotopes.isChecked() and self.check_isotopes.isEnabled()

    def updatePreview(self) -> None:
        self.lineedit_preview.setText(
            os.path.basename(
                self.generatePath(
                    isotope=self.isotope if self.exportAllIsotopes() else ""
                )
            )
        )

    def generatePath(self, path: str, isotope: str = "") -> str:
        base, ext = os.path.splitext(path)
        isotope = isotope.replace(os.path.sep, "_")
        return f"{base}{'_' if isotope else ''}{isotope}{ext}"

    def generatePaths(self, path: str) -> List[Tuple[str, str]]:
        if self.exportAllIsotopes():
            return [(self.generatePath(path, i), i) for i in self.laser.isotopes]
        else:
            return [(self.generatePath(path), self.isotope)]

    def accept(self) -> None:
        paths = self.generatePaths()
        prompt = OverwriteFilePrompt()
        paths = [p for p, _ in paths if prompt.promptOverwrite(p)]

        if len(paths) != 0:
            for p in paths:
                self.export(p)


class CsvExportDialog(ExportDialog):
    def __init__(
        self, laser: Laser, current_isotope: str, parent: QtWidgets.QWidget = None
    ):
        super().__init__(laser, current_isotope, CsvExportOptions(), parent)

    def export(
        self,
        path: str,
        calibrate: bool,
        view_limits: Tuple[float, float, float, float] = None,
    ) -> None:
        paths = self._generate_paths(path)
        kwargs = {"calibrate": calibrate, "flat": True}
        if self.dlg.options.trimmedChecked():
            kwargs["extent"] = view_limits
        for path, isotope, _ in paths:
            io.csv.save(path, self.laser.get(isotope, **kwargs))


class PngExportDialog(ExportDialog):
    def __init__(
        self,
        laser: Laser,
        current_isotope: str,
        canvas: LaserCanvas,
        parent: QtWidgets.QWidget = None,
    ):
        self.canvas = canvas
        view_limits = canvas.view_limits
        if view_limits is not None:
            x = view_limits[1] - view_limits[0]
            y = view_limits[3] - view_limits[2]
            imagesize = (1280, int(1280 * x / y)) if x > y else (int(800 * y / x), 800)
        else:
            imagesize = (1280, 800)
        super().__init__(
            laser, current_isotope, PngExportOptions(imagesize=imagesize), parent
        )

    def export(self, path: str) -> None:
        paths = self._generate_paths(path)
        old_size = self.canvas.figure.get_size_inches()
        size = self.options.imagesize()
        dpi = self.canvas.figure.get_dpi()
        self.canvas.figure.set_size_inches(size[0] / dpi, size[1] / dpi)

        for path, isotope, _ in paths:
            self.canvas.drawLaser(self.laser, isotope)
            self.canvas.figure.savefig(path, transparent=True, frameon=False)

        self.canvas.figure.set_size_inches(*old_size)
        self.canvas.drawLaser(self.laser, self.current_isotope)
        self.canvas.draw()


class CsvExportDialog(ExportDialog):
    def __init__(
        self, laser: Laser, current_isotope: str, parent: QtWidgets.QWidget = None
    ):
        super().__init__(laser, current_isotope, CsvExportOptions(), parent)

    def export(
        self,
        path: str,
        calibrate: bool,
        view_limits: Tuple[float, float, float, float] = None,
    ) -> None:
        paths = self._generate_paths(path)
        kwargs = {"calibrate": calibrate, "flat": True}
        if self.dlg.options.trimmedChecked():
            kwargs["extent"] = view_limits
        for path, isotope, _ in paths:
            io.csv.save(path, self.laser.get(isotope, **kwargs))
