from pathlib import Path
import logging

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib import io
from pewlib.laser import Laser

from pewpew.widgets.prompts import OverwriteFilePrompt
from pewpew.widgets.views import _ViewWidget

from typing import List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


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
        self.lineedits = [QtWidgets.QLineEdit(str(dim)) for dim in spacing]
        for le in self.lineedits:
            le.setValidator(QtGui.QDoubleValidator(-1e9, 1e9, 4))
            le.textEdited.connect(self.inputChanged)
        self.lineedits[0].setEnabled(False)  # X
        self.lineedits[1].setEnabled(False)  # Y

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Spacing:"), 0)
        layout.addWidget(self.lineedits[0], 0)  # X
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.lineedits[1], 0)  # Y
        layout.addWidget(QtWidgets.QLabel("x"), 0, QtCore.Qt.AlignCenter)
        layout.addWidget(self.lineedits[2], 0)  # Z
        layout.addStretch(1)

        self.setLayout(layout)

    def isComplete(self) -> bool:
        return all(le.hasAcceptableInput() for le in self.lineedits)

    def spacing(self) -> Tuple[float, float, float]:
        return tuple(float(le.text()) for le in self.lineedits)  # type: ignore


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
        else:  # pragma: no cover
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
    invalid_chars = '<>:"/\\|?*'
    invalid_map = str.maketrans(invalid_chars, "_" * len(invalid_chars))

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

        filename_regexp = QtCore.QRegExp(f"[^{self.invalid_chars}]+")
        self.lineedit_filename.setValidator(QtGui.QRegExpValidator(filename_regexp))
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
        if not Path(self.lineedit_directory.text()).exists():
            return False
        # if self.options.indexForExt(Path(self.lineedit_filename.text()).suffix) == -1:
        #     print("indexForExt")
        #     return False
        if not self.options.isComplete():
            return False
        return True

    def validate(self) -> None:
        ok = self.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        ok.setEnabled(self.isComplete())

    def updatePreview(self) -> None:
        self.lineedit_preview.setText(self.lineedit_filename.text())

    def filenameChanged(self, filename: str) -> None:
        self.options.setCurrentExt(Path(filename).suffix.lower())
        self.updatePreview()

    def typeChanged(self, index: int) -> None:
        # Update name of file
        path = Path(self.lineedit_filename.text())
        if path.suffix != "":
            self.lineedit_filename.setText(
                str(path.with_suffix(self.options.currentExt()))
            )

        self.adjustSize()

    def selectDirectory(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(
            self, "Select Directory", self.lineedit_directory.text()
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dlg.fileSelected.connect(self.lineedit_directory.setText)
        dlg.open()
        return dlg


class ExportDialog(_ExportDialogBase):
    def __init__(self, widget: _ViewWidget, parent: QtWidgets.QWidget = None):
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

        self.check_export_all = QtWidgets.QCheckBox("Export all elements.")
        self.check_export_all.setToolTip(
            "Export all elements for the current image.\n"
            "The filename will be appended with the elements name."
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
        path = self.widget.laserFilePath(ext=".npz").resolve()
        self.lineedit_directory.setText(str(path.parent))
        self.lineedit_filename.setText(str(path.name).translate(self.invalid_map))
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

    def isComplete(self) -> bool:
        suffix = Path(self.lineedit_filename.text()).suffix
        if suffix != "" and self.options.indexForExt(suffix) == -1:
            return False
        return super().isComplete()

    def updatePreview(self) -> None:
        path = Path(self.lineedit_filename.text())
        if self.isExportAll():
            path = path.with_name(path.stem + "_<element>")
        if self.isExportLayers():
            path = path.with_name(path.stem + "_<layer#>")

        if path.suffix == "":
            path = path.with_suffix(self.options.currentExt())
        self.lineedit_preview.setText(str(path))

    def typeChanged(self, index: int) -> None:
        super().typeChanged(index)
        # Enable or disable checks
        self.check_calibrate.setEnabled(self.allowCalibrate())
        self.check_export_all.setEnabled(self.allowExportAll())
        self.check_export_layers.setEnabled(self.allowExportLayers())
        self.updatePreview()

    def getPath(self) -> Path:
        path = Path(self.lineedit_filename.text())
        if path.suffix == "":
            path = path.with_suffix(self.options.currentExt())
        return Path(self.lineedit_directory.text()).joinpath(path)

    def getPathForElement(self, path: Path, element: str) -> Path:
        return path.with_name(
            path.stem + "_" + element.translate(self.invalid_map) + path.suffix
        )

    def getPathForLayer(self, path: Path, layer: int) -> Path:
        return path.with_name(path.stem + "_layer" + str(layer) + path.suffix)

    def generatePaths(self, laser: Laser) -> List[Tuple[Path, str, Optional[int]]]:
        paths: List[Tuple[Path, str, Optional[int]]] = [
            (self.getPath(), self.widget.combo_element.currentText(), None)
        ]
        if self.isExportAll():
            paths = [
                (self.getPathForElement(p, i), i, None)
                for i in laser.elements
                for (p, _, _) in paths
            ]
        if self.isExportLayers():
            paths = [
                (self.getPathForLayer(p, j), i, j)
                for j in range(0, laser.layers)
                for (p, i, _) in paths
            ]

        return [p for p in paths if p[0] != ""]

    def export(self, path: Path, element: str, layer: Optional[int], widget: _ViewWidget) -> None:
        option = self.options.currentOption()

        if option.ext == ".csv":
            kwargs = {"calibrate": self.isCalibrate(), "flat": True}
            if element in widget.laser.elements:
                data = widget.laser.get(element, layer=layer, **kwargs)
                io.textimage.save(path, data)

        elif option.ext == ".png":
            if element in widget.laser.elements:

                self.widget.graphics.drawLaser(
                    self.widget.laser,
                    element,
                    layer=self.widget.current_layer,
                )
                if self.widget.graphics.widget is not None:
                    self.widet.graphics.widget.imageChanged(
                        self.widget.graphics.image, self.widget.graphics.data
                    )

                self.widget.graphics.updateForeground()
                self.widget.graphics.invalidateScene()
                if option.raw():
                    self.widget.graphics.image.image.save(str(path.absolute()))
                else:
                    pixmap = QtGui.QPixmap(self.widget.graphics.viewport().size())
                    painter = QtGui.QPainter(pixmap)
                    self.widget.graphics.render(painter)
                    painter.end()
                    pixmap.save(str(path.absolute()))

        elif option.ext == ".vti":
            spacing = option.spacing()
            # Last axis (z) is negative for layer order
            spacing = spacing[0], spacing[1], -spacing[2]
            data = widget.laser.get(calibrate=self.isCalibrate())
            io.vtk.save(path, data, spacing)

        elif option.ext == ".npz":  # npz
            io.npz.save(path, widget.laser)

        else:
            raise ValueError(f"Unable to export file as '{option.ext}'.")

        logger.info(f"Exported {widget.laser.info['Name']} to {path.name}.")

    def accept(self) -> None:
        paths = self.generatePaths(self.widget.laser)
        prompt = OverwriteFilePrompt()
        paths = [p for p in paths if prompt.promptOverwrite(str(p[0].resolve()))]

        if len(paths) == 0:
            return

        try:
            for path, element, layer in paths:
                self.export(path, element, layer, self.widget)
        except Exception as e:  # pragma: no cover
            logger.exception(e)
            QtWidgets.QMessageBox.critical(self, "Unable to Export!", str(e))
            return

        self.widget.refresh()
        super().accept()


class ExportAllDialog(ExportDialog):
    def __init__(self, widgets: List[_ViewWidget], parent: QtWidgets.QWidget = None):
        unique: Set[str] = set()
        for widget in widgets:
            unique.update(widget.laser.elements)
        elements = sorted(unique)

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.addItems(elements)
        self.lineedit_prefix = QtWidgets.QLineEdit("")
        prefix_regexp = QtCore.QRegExp(f"[^{self.invalid_chars}]+")
        self.lineedit_prefix.setValidator(QtGui.QRegExpValidator(prefix_regexp))
        self.lineedit_prefix.textChanged.connect(self.updatePreview)

        super().__init__(widgets[0], parent)
        self.setWindowTitle("Export All")
        self.widgets = widgets

        # Adjust widgets for all
        self.lineedit_filename.setText("<name>.npz")
        label = self.layout_form.labelForField(self.lineedit_filename)
        label.setText("Prefix:")
        self.layout_form.replaceWidget(self.lineedit_filename, self.lineedit_prefix)

        self.layout_form.addRow("Element:", self.combo_element)

        self.check_export_all.stateChanged.connect(self.showElements)
        self.showElements()

    def showElements(self) -> None:
        self.combo_element.setEnabled(self.allowExportAll() and not self.isExportAll())

    def typeChanged(self, index: int) -> None:
        super().typeChanged(index)
        self.showElements()

    def updatePreview(self) -> None:
        path = Path(self.lineedit_filename.text())
        prefix = self.lineedit_prefix.text()
        if prefix != "":
            path = path.with_name(prefix + "_" + path.name)
        if self.isExportAll():
            path = path.with_name(path.stem + "_<element>" + path.suffix)
        self.lineedit_preview.setText(str(path))

    def getPath(self, name: str) -> Path:
        path = (
            Path(self.lineedit_directory.text())
            .joinpath(name)
            .with_suffix(Path(self.lineedit_filename.text()).suffix)
        )
        prefix = self.lineedit_prefix.text()
        if prefix != "":
            path = path.with_name(prefix + "_" + path.name)
        return path

    def generatePaths(self, laser: Laser) -> List[Tuple[Path, str, Optional[int]]]:
        paths: List[Tuple[Path, str, Optional[int]]] = [
            (
                self.getPath(laser.info["Name"]),
                self.widget.combo_element.currentText(),
                None,
            )
        ]
        if self.isExportAll():
            paths = [
                (self.getPathForElement(p, i), i, None)
                for i in laser.elements
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
            all_paths.append(
                [p for p in paths if prompt.promptOverwrite(str(p[0].resolve()))]
            )

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
                for path, element, layer in paths:
                    dlg.setValue(exported)
                    if dlg.wasCanceled():
                        break
                    self.export(path, element, layer, widget)
                    exported += 1
        except Exception as e:  # pragma: no cover
            logger.exception(e)
            QtWidgets.QMessageBox.critical(self, "Unable to Export!", str(e))
            dlg.cancel()
            return

        dlg.setValue(count)
        QtWidgets.QDialog.accept(self)
