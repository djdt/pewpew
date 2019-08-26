from PySide2 import QtGui, QtWidgets

import os.path

from laserlib import io
from laserlib.laser import Laser

from pewpew.widgets.canvases import BasicCanvas
from pewpew.widgets.prompts import OverwriteFilePrompt

from typing import List, Tuple


class CSVExportOptions(QtWidgets.QGroupBox):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("CSV Options", parent)
        self.check_trim = QtWidgets.QCheckBox("Trim data to view.")
        self.check_trim.setChecked(True)

        options_layout = QtWidgets.QVBoxLayout()
        options_layout.addWidget(self.check_trim)
        self.setLayout(options_layout)

    def trimmedChecked(self) -> bool:
        return self.check_trim.isChecked()


class PNGExportOptions(QtWidgets.QGroupBox):
    def __init__(
        self, imagesize: Tuple[int, int] = (1280, 800), parent: QtWidgets.QWidget = None
    ):
        super().__init__("PNG Options", parent)

        self.linedit_size_x = QtWidgets.QLineEdit(str(imagesize[0]))
        self.linedit_size_x.setValidator(QtGui.QIntValidator(0, 9999))
        self.linedit_size_y = QtWidgets.QLineEdit(str(imagesize[1]))
        self.linedit_size_y.setValidator(QtGui.QIntValidator(0, 9999))

        layout_size = QtWidgets.QHBoxLayout()
        layout_size.addWidget(QtWidgets.QLabel("Size:"))
        layout_size.addWidget(self.linedit_size_x)
        layout_size.addWidget(self.linedit_size_y)

        options_layout = QtWidgets.QVBoxLayout()
        options_layout.addLayout(layout_size)
        self.setLayout(options_layout)

    def imagesize(self) -> Tuple[int, int]:
        return (int(self.linedit_size_x.text()), int(self.linedit_size_y.text()))


class ExportAllOptions(QtWidgets.QStackedWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.default = QtWidgets.QWidget()
        self.csv = CSVExportOptions()
        self.png = PNGExportOptions()

        self.addWidget(self.default)
        self.addWidget(self.csv)
        self.addWidget(self.png)

    def update(self, ext: str) -> None:
        if ext == ".csv":
            self.setCurrentWidget(self.csv)
        elif ext == ".png":
            self.setCurrentWidget(self.png)
        else:
            self.setCurrentWidget(self.default)


class ExportDialog(QtWidgets.QDialog):
    def __init__(
        self,
        laser: Laser,
        current_isotope: str,
        options: QtWidgets.QWidget,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.isotope = current_isotope
        self.laser = laser
        self.options = options

        self.check_isotopes = QtWidgets.QCheckBox("Export all isotopes.")
        if len(laser.isotopes) < 2:
            self.check_isotopes.setEnabled(False)
        self.check_isotopes.stateChanged.connect(self.updatePreview)

        self.lineedit_preview = QtWidgets.QLineEdit()
        self.lineedit_preview.setEnabled(False)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.layoutWidgets()
        self._update()

    def layoutWidgets(self) -> None:
        layout_preview = QtWidgets.QHBoxLayout()
        layout_preview.addWidget(QtWidgets.QLabel("Preview:"))
        layout_preview.addWidget(self.lineedit_preview)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addWidget(self.options)
        layout_main.addWidget(self.check_isotopes)
        layout_main.addLayout(layout_preview)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

    def updatePreview(self) -> None:
        self.lineedit_preview.setText(
            os.path.basename(
                self._generate_path(
                    isotope=self.isotope
                    if self.check_isotopes.isChecked()
                    and self.check_isotopes.isEnabled()
                    else None
                )
            )
        )

    def generate_path(self, path: str, isotope: str = None, layer: int = None) -> str:
        base, ext = os.path.splitext(path)
        if isotope is not None:
            if os.path.sep in isotope:
                isotope = isotope.replace(os.path.sep, "_")
            base += f"_{isotope}"
        if layer is not None:
            base += f"_{layer}"

        return base + ext

    def generatePaths(self, path: str) -> List[Tuple[str, str, int]]:
        paths = []
        if self.check_isotopes.isChecked() and self.check_isotopes.isEnabled():
            prompt = OverwriteFilePrompt(parent=self)
            for isotope in self.laser.isotopes:
                p = self._generate_path(path, isotope)
                if p == "" or not prompt.promptOverwrite(path):
                    continue
                paths.append((p, isotope, -1))
        else:
            prompt = OverwriteFilePrompt(show_all_buttons=False, parent=self)
            p = self._generate_path(path)
            if p != "" and prompt.promptOverwrite(path):
                paths.append((p, self.isotope, -1))
        return paths

    def export(self, path: str) -> bool:
        return NotImplementedError


class CSVExportDialog(ExportDialog):
    def __init__(
        self,
        laser: Laser,
        current_isotope: str,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(laser, current_isotope, CSVExportOptions(), parent)

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


class PNGExportDialog(ExportDialog):
    def __init__(
        self,
        laser: Laser,
        current_isotope: str,
        view_limits: Tuple[float, float, float, float] = None,
        parent: QtWidgets.QWidget = None,
    ):
        imagesize = (1280, 800)
        if view_limits is not None:
            x = view_limits[1] - view_limits[0]
            y = view_limits[3] - view_limits[2]
            imagesize = (1280, int(1280 * x / y)) if x > y else (int(800 * y / x), 800)
        super().__init__(
            laser, current_isotope, PNGExportOptions(imagesize=imagesize), parent
        )

    def export(self, path: str, canvas: BasicCanvas) -> None:
        paths = self._generate_paths(path)
        old_size = canvas.figure.get_size_inches()
        size = self.options.imagesize()
        dpi = canvas.figure.get_dpi()
        canvas.figure.set_size_inches(size[0] / dpi, size[1] / dpi)

        for path, isotope, _ in paths:
            canvas.drawLaser(self.laser, isotope)
            canvas.figure.savefig(path, transparent=True, frameon=False)

        canvas.figure.set_size_inches(*old_size)
        canvas.drawLaser(self.laser, self.current_isotope)
        canvas.draw()


class ExportAllDialog(ExportDialog):
    FORMATS = {
        "CSV File": ".csv",
        "Numpy Archive": ".npz",
        "PNG Image": ".png",
        "VTK Image": ".vti",
    }

    def __init__(
        self,
        path: str,
        name: str,
        isotopes: List[str],
        layers: int,
        parent: QtWidgets.QWidget = None,
    ):
        self.name = name
        layout_all = QtWidgets.QFormLayout()
        self.lineedit_prefix = QtWidgets.QLineEdit()
        self.lineedit_prefix.textChanged.connect(self._update)

        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.addItems(isotopes)
        self.combo_isotopes.currentIndexChanged.connect(self._update)

        self.combo_formats = QtWidgets.QComboBox()
        self.combo_formats.addItems(list(ExportAllDialog.FORMATS.keys()))
        self.combo_formats.currentIndexChanged.connect(self._update)

        super().__init__(
            path, isotopes[0], len(isotopes), layers, ExportAllOptions(), parent
        )

        layout_all.addRow("Prefix:", self.lineedit_prefix)
        layout_all.addRow("Isotope:", self.combo_isotopes)
        layout_all.addRow("Format:", self.combo_formats)

        self.layout().insertLayout(0, layout_all)

    def _update(self) -> None:
        ext = ExportAllDialog.FORMATS[self.combo_formats.currentText()]
        self.options.update(ext)
        if ext in [".npz", ".vti"]:
            self.check_isotopes.setEnabled(False)
            self.check_isotopes.setChecked(True)
        elif not self.check_isotopes.isEnabled():
            self.check_isotopes.setEnabled(True)
            self.check_isotopes.setChecked(False)
        self.combo_isotopes.setEnabled(not self.check_isotopes.isChecked())

        self.lineedit_preview.setText(
            os.path.basename(
                self._generate_path(
                    isotope=self.combo_isotopes.currentText()
                    if self.check_isotopes.isChecked()
                    and self.check_isotopes.isEnabled()
                    else None
                )
            )
        )

    def _get_isotope(self) -> str:
        return self.combo_isotopes.currentText()

    def _generate_path(
        self, laser: Laser = None, isotope: str = None, layer: int = None
    ) -> str:
        if laser is not None:
            if isotope is not None and isotope not in laser.isotopes:
                return ""
            elif self._get_isotope() not in laser.isotopes:
                return ""
        name = self.name if laser is None else laser.name

        prefix = self.lineedit_prefix.text()
        path = os.path.join(self.path, prefix + "_" + name if prefix != "" else name)
        ext = ExportAllDialog.FORMATS[self.combo_formats.currentText()]
        if isotope is not None:
            path += f"_{isotope}"
        if layer is not None:
            path += f"_{layer}"
        return path + ext
