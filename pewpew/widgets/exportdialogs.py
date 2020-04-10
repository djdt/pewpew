import sys
import os.path

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.laser import Laser

from pewpew.widgets.canvases import LaserCanvas
from pewpew.widgets.prompts import OverwriteFilePrompt
from pewpew.widgets.laser import LaserWidget

from typing import List, Set, Tuple


class OptionsBox(QtWidgets.QGroupBox):
    inputChanged = QtCore.Signal()

    def __init__(
        self,
        filetype: str,
        ext: str,
        visible: bool = False,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__("Format Options", parent)
        self.filetype = filetype
        self.ext = ext
        self.visible = visible

    # Because you can't hook up signals with different no. of params
    def isComplete(self) -> bool:
        return True


class PngOptionsBox(OptionsBox):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("PNG Images", ".png", visible=True, parent=parent)
        self.check_raw = QtWidgets.QCheckBox("Save raw image data.")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.check_raw)
        self.setLayout(layout)

    def raw(self) -> bool:
        return self.check_raw.isChecked()


class VtiOptionsBox(OptionsBox):
    def __init__(
        self, spacing: Tuple[float, float, float], parent: QtWidgets.QWidget = None
    ):
        super().__init__("VTK Images", ".vti", visible=True, parent=parent)
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
        return tuple(float(le.text()) for le in self.linedits)  # type: ignore


class _ExportOptionsStack(QtWidgets.QStackedWidget):
    def sizeHint(self) -> QtCore.QSize:
        sizes = [self.widget(i).sizeHint() for i in range(0, self.count())]
        return QtCore.QSize(
            max(s.width() for s in sizes), max(s.height() for s in sizes)
        )


class ExportOptions(QtWidgets.QWidget):
    inputChanged = QtCore.Signal()
    currentIndexChanged = QtCore.Signal(int)

    def __init__(
        self, options: List[OptionsBox] = None, parent: QtWidgets.QWidget = None
    ):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding
        )
        self.stack = _ExportOptionsStack()
        self.stack.currentChanged.connect(self.currentIndexChanged)

        self.combo = QtWidgets.QComboBox()
        self.combo.currentIndexChanged.connect(self.stack.setCurrentIndex)

        if options is not None:
            for option in options:
                self.addOption(option)

        layout_form = QtWidgets.QFormLayout()
        if self.count() < 2:
            self.combo.setVisible(False)
        else:
            layout_form.addRow("Type:", self.combo)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_form)
        layout.addWidget(self.stack)
        self.setLayout(layout)

    def addOption(self, option: OptionsBox) -> int:
        index = self.stack.addWidget(option)
        self.stack.widget(index).inputChanged.connect(self.inputChanged)

        item = f"{option.filetype} ({option.ext})"
        self.combo.insertItem(index, item)
        return index

    def currentOption(self) -> OptionsBox:
        return self.stack.currentWidget()

    def count(self) -> int:
        return self.stack.count()

    def currentExt(self) -> str:
        return self.stack.currentWidget().ext

    def currentIndex(self) -> int:
        return self.stack.currentIndex()

    def setCurrentExt(self, ext: str) -> None:
        index = self.indexForExt(ext)
        if index != -1:
            self.setCurrentIndex(index)

    def setCurrentIndex(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.combo.setCurrentIndex(index)

        self.stack.setVisible(self.stack.currentWidget().visible)

    def indexForExt(self, ext: str) -> int:
        for i in range(self.stack.count()):
            if self.stack.widget(i).ext == ext:
                return i
        return -1

    def isComplete(self, current_only: bool = True) -> bool:
        indicies = [self.currentIndex()] if current_only else range(0, self.count())
        return all(self.stack.widget(i).isComplete() for i in indicies)


class _ExportDialogBase(QtWidgets.QDialog):
    def __init__(self, options: List[OptionsBox], parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Export")

        self.lineedit_directory = QtWidgets.QLineEdit()
        self.lineedit_directory.setMinimumWidth(300)
        self.lineedit_directory.setClearButtonEnabled(True)
        self.lineedit_directory.textChanged.connect(self.validate)

        icon = QtGui.QIcon.fromTheme("document-open-folder")
        self.button_directory = QtWidgets.QPushButton(
            icon, "Open" if icon.isNull() else ""
        )
        self.button_directory.clicked.connect(self.selectDirectory)
        self.lineedit_filename = QtWidgets.QLineEdit()
        self.lineedit_filename.textChanged.connect(self.filenameChanged)
        self.lineedit_filename.textChanged.connect(self.validate)

        self.lineedit_preview = QtWidgets.QLineEdit()
        self.lineedit_preview.setEnabled(False)

        self.options = ExportOptions(options=options)
        self.options.inputChanged.connect(self.validate)
        self.options.currentIndexChanged.connect(self.typeChanged)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout_directory = QtWidgets.QHBoxLayout()
        layout_directory.addWidget(self.lineedit_directory)
        layout_directory.addWidget(self.button_directory)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout_form = QtWidgets.QFormLayout()
        self.layout_form.addRow("Directory:", layout_directory)
        self.layout_form.addRow("Filename:", self.lineedit_filename)
        self.layout_form.addRow("Preview:", self.lineedit_preview)

        self.layout.addLayout(self.layout_form)
        self.layout.addWidget(self.options, 1, QtCore.Qt.AlignTop)
        self.layout.addWidget(self.button_box)
        self.setLayout(self.layout)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 200)

    def isComplete(self) -> bool:
        if not os.path.exists(self.lineedit_directory.text()):
            return False
        filename = self.lineedit_filename.text()
        if filename == "":
            return False
        if self.options.indexForExt(os.path.splitext(filename)[1]) == -1:
            return False
        if not self.options.isComplete():
            return False
        return True

    def validate(self) -> None:
        ok = self.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        ok.setEnabled(self.isComplete())

    def updatePreview(self) -> None:
        self.lineedit_preview.setText(self.lineedit_filename.text())

    def filenameChanged(self, filename: str) -> None:
        _, ext = os.path.splitext(filename.lower())
        index = self.options.indexForExt(ext)
        if index == -1:
            return
        self.options.setCurrentIndex(index)
        self.updatePreview()

    def typeChanged(self, index: int) -> None:
        # Update name of file
        base, ext = os.path.splitext(self.lineedit_filename.text())
        if ext != "":
            ext = self.options.currentExt()
        self.lineedit_filename.setText(base + ext)

        self.adjustSize()

    def selectDirectory(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(
            self, "Select Directory", self.lineedit_directory.text()
        )
        if sys.platform != "win32":  # TODO wait for fix
            dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.lineedit_directory.setText)
        dlg.open()
        return dlg


class ExportDialog(_ExportDialogBase):
    def __init__(self, widget: LaserWidget, parent: QtWidgets.QWidget = None):
        spacing = (
            widget.laser.config.get_pixel_width(),
            widget.laser.config.get_pixel_height(),
            widget.laser.config.spotsize / 2.0,
        )
        options = [
            OptionsBox("Numpy Archives", ".npz"),
            OptionsBox("CSV Document", ".csv"),
            PngOptionsBox(),
            VtiOptionsBox(spacing),
        ]
        super().__init__(options, parent)
        self.widget = widget

        self.check_calibrate = QtWidgets.QCheckBox("Calibrate data.")
        self.check_calibrate.setChecked(True)
        self.check_calibrate.setToolTip("Calibrate the data before exporting.")

        self.check_export_all = QtWidgets.QCheckBox("Export all isotopes.")
        self.check_export_all.setToolTip(
            "Export all isotopes for the current image.\n"
            "The filename will be appended with the isotopes name."
        )
        self.check_export_all.clicked.connect(self.updatePreview)

        self.check_export_layers = QtWidgets.QCheckBox("Export all layers.")
        self.check_export_layers.setToolTip(
            "Export layers individually.\n"
            "The filename will be appended with the layer number."
        )
        self.check_export_layers.clicked.connect(self.updatePreview)
        self.check_export_layers.setEnabled(widget.is_srr)
        self.check_export_layers.setVisible(widget.is_srr)

        self.layout.insertWidget(2, self.check_calibrate)
        self.layout.insertWidget(3, self.check_export_all)
        self.layout.insertWidget(4, self.check_export_layers)

        # A default path
        path = os.path.join(
            os.path.dirname(self.widget.laser.path), self.widget.laser.name + ".npz"
        )
        self.lineedit_directory.setText(os.path.dirname(path))
        self.lineedit_filename.setText(os.path.basename(path))
        self.typeChanged(0)

    def allowCalibrate(self) -> bool:
        return self.options.currentExt() != ".npz"

    def allowExportAll(self) -> bool:
        return self.options.currentExt() not in [".npz", ".vti"]

    def allowExportLayers(self) -> bool:
        return self.options.currentExt() not in [".npz", ".png", ".vti"]

    def isCalibrate(self) -> bool:
        return self.check_calibrate.isChecked() and self.check_calibrate.isEnabled()

    def isExportAll(self) -> bool:
        return self.check_export_all.isChecked() and self.check_export_all.isEnabled()

    def isExportLayers(self) -> bool:
        return (
            self.check_export_layers.isChecked()
            and self.check_export_layers.isEnabled()
        )

    def updatePreview(self) -> None:
        base, ext = os.path.splitext(self.lineedit_filename.text())
        if self.isExportAll():
            base += "_<ISOTOPE>"
        if self.isExportLayers():
            base += "_layer<#>"
        self.lineedit_preview.setText(base + ext)

    def typeChanged(self, index: int) -> None:
        super().typeChanged(index)
        # Enable or disable checks
        self.check_calibrate.setEnabled(self.allowCalibrate())
        self.check_export_all.setEnabled(self.allowExportAll())
        self.check_export_layers.setEnabled(self.allowExportLayers())
        self.updatePreview()

    def getPath(self) -> str:
        return os.path.join(
            self.lineedit_directory.text(), self.lineedit_filename.text()
        )

    def getPathForIsotope(self, path: str, isotope: str) -> str:
        base, ext = os.path.splitext(path)
        isotope = isotope.replace(os.path.sep, "_")
        return f"{base}_{isotope}{ext}"

    def getPathForLayer(self, path: str, layer: int) -> str:
        base, ext = os.path.splitext(path)
        return f"{base}_layer{layer}{ext}"

    def generatePaths(self, laser: Laser) -> List[Tuple[str, str, int]]:
        paths: List[Tuple[str, str, int]] = [
            (self.getPath(), self.widget.combo_isotopes.currentText(), None)
        ]
        if self.isExportAll():
            paths = [
                (self.getPathForIsotope(p, i), i, None)
                for i in laser.isotopes
                for (p, _, _) in paths
            ]
        if self.isExportLayers():
            paths = [
                (self.getPathForLayer(p, j), i, j)
                for j in range(0, laser.layers)
                for (p, i, _) in paths
            ]

        return [p for p in paths if p[0] != ""]

    def export(self, path: str, isotope: str, layer: int, widget: LaserWidget) -> None:
        option = self.options.currentOption()

        if option.ext == ".csv":
            kwargs = {"calibrate": self.isCalibrate(), "flat": True}
            if isotope in widget.laser.isotopes:
                data = widget.laser.get(isotope, layer=layer, **kwargs)
                io.csv.save(path, data)

        elif option.ext == ".png":
            canvas = LaserCanvas(widget.canvas.viewoptions, self)
            if isotope in widget.laser.isotopes:
                canvas.drawLaser(widget.laser, isotope)
                if option.raw():
                    canvas.saveRawImage(path)
                else:
                    canvas.view_limits = widget.canvas.view_limits
                    canvas.figure.set_size_inches(
                        widget.canvas.figure.get_size_inches()
                    )
                    canvas.figure.savefig(
                        path,
                        dpi=300,
                        bbox_inches="tight",
                        transparent=True,
                        facecolor=None,
                    )
            canvas.close()

        elif option.ext == ".vti":
            spacing = option.spacing()
            data = widget.laser.get(calibrate=self.isCalibrate())
            io.vtk.save(path, data, spacing)

        elif option.ext == ".npz":  # npz
            io.npz.save(path, [widget.laser])

        else:
            raise io.error.PewException(f"Unable to export file as '{option.ext}'.")

    def accept(self) -> None:
        paths = self.generatePaths(self.widget.laser)
        prompt = OverwriteFilePrompt()
        paths = [p for p in paths if prompt.promptOverwrite(p[0])]

        if len(paths) == 0:
            return

        try:
            for path, isotope, layer in paths:
                self.export(path, isotope, layer, self.widget)
        except io.error.PewException as e:
            QtWidgets.QMessageBox.critical(self, "Unable to Export!", str(e))
            return

        super().accept()


class ExportAllDialog(ExportDialog):
    def __init__(self, widgets: List[LaserWidget], parent: QtWidgets.QWidget = None):
        unique: Set[str] = set()
        for widget in widgets:
            unique.update(widget.laser.isotopes)
        isotopes = sorted(unique)

        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.addItems(isotopes)
        self.lineedit_prefix = QtWidgets.QLineEdit("")
        self.lineedit_prefix.textChanged.connect(self.updatePreview)

        super().__init__(widgets[0], parent)
        self.setWindowTitle("Export All")
        self.widgets = widgets

        # Adjust widgets for all
        self.lineedit_filename.setText("<NAME>.npz")
        label = self.layout_form.labelForField(self.lineedit_filename)
        label.setText("Prefix:")
        self.layout_form.replaceWidget(self.lineedit_filename, self.lineedit_prefix)

        self.layout_form.addRow("Isotope:", self.combo_isotopes)

        self.check_export_all.stateChanged.connect(self.showIsotopes)
        self.showIsotopes()

    def showIsotopes(self) -> None:
        self.combo_isotopes.setEnabled(self.allowExportAll() and not self.isExportAll())

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

    def getPath(self, name: str = None) -> str:
        _, ext = os.path.splitext(self.lineedit_filename.text())
        prefix = self.lineedit_prefix.text()
        if prefix != "":
            prefix += "_"
        return os.path.join(self.lineedit_directory.text(), f"{prefix}{name}{ext}")

    def generatePaths(self, laser: Laser) -> List[Tuple[str, str, int]]:
        paths: List[Tuple[str, str, int]] = [
            (self.getPath(laser.name), self.widget.combo_isotopes.currentText(), None)
        ]
        if self.isExportAll():
            paths = [
                (self.getPathForIsotope(p, i), i, None)
                for i in laser.isotopes
                for (p, _, _) in paths
            ]
        if self.isExportLayers():
            paths = [
                (self.getPathForLayer(p, j), i, j)
                for j in range(0, laser.layers)
                for (p, i, _) in paths
            ]

        return [p for p in paths if p[0] != ""]

    def accept(self) -> None:
        all_paths = []
        prompt = OverwriteFilePrompt()

        for widget in self.widgets:
            paths = self.generatePaths(widget.laser)
            all_paths.append([p for p in paths if prompt.promptOverwrite(p[0])])

        if all(len(paths) == 0 for paths in all_paths):
            return

        count = sum([len(p) for p in all_paths])
        exported = 0
        dlg = QtWidgets.QProgressDialog("Exporting...", "Abort", 0, count, self)
        dlg.setWindowTitle("Exporting Data")
        dlg.open()

        try:
            for paths, widget in zip(all_paths, self.widgets):
                widget.setActive()
                for path, isotope, layer in paths:
                    dlg.setValue(exported)
                    if dlg.wasCanceled():
                        break
                    self.export(path, isotope, layer, widget)
                    exported += 1
        except io.error.PewException as e:
            QtWidgets.QMessageBox.critical(self, "Unable to Export!", str(e))
            dlg.cancel()
            return

        dlg.setValue(count)
        QtWidgets.QDialog.accept(self)
