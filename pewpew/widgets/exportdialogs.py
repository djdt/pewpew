import logging
from pathlib import Path
from typing import Set

from pewlib import io
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.graphics.export import generate_laser_image, generate_rgb_laser_image
from pewpew.graphics.imageitems import LaserImageItem, RGBLaserImageItem
from pewpew.graphics.options import GraphicsOptions
from pewpew.widgets.prompts import OverwriteFilePrompt

logger = logging.getLogger(__name__)


class OptionsBox(QtWidgets.QGroupBox):
    inputChanged = QtCore.Signal()

    def __init__(
        self,
        filetype: str,
        ext: str,
        visible: bool = False,
        allow_export_all: bool = True,
        allow_calibrate: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Format Options", parent)
        self.filetype = filetype
        self.ext = ext
        self.visible = visible

        self.allow_export_all = allow_export_all
        self.allow_calibrate = allow_calibrate

    def allowCalibrate(self) -> bool:
        return self.allow_calibrate

    def allowExportAll(self) -> bool:
        return self.allow_export_all

    def isComplete(self) -> bool:
        return True


class PngOptionsBox(OptionsBox):
    item_positions = {
        "Top Left": QtCore.Qt.AlignmentFlag.AlignTop
        | QtCore.Qt.AlignmentFlag.AlignLeft,
        "Top Right": QtCore.Qt.AlignmentFlag.AlignTop
        | QtCore.Qt.AlignmentFlag.AlignRight,
        "Bottom Left": QtCore.Qt.AlignmentFlag.AlignBottom
        | QtCore.Qt.AlignmentFlag.AlignLeft,
        "Bottom Right": QtCore.Qt.AlignmentFlag.AlignBottom
        | QtCore.Qt.AlignmentFlag.AlignRight,
        "Off": None,
    }

    def __init__(self, size: QtCore.QSize, parent: QtWidgets.QWidget | None = None):
        super().__init__("PNG Images", ".png", visible=True)

        self.base_size = size
        self.label_x = QtWidgets.QLabel()
        self.label_y = QtWidgets.QLabel()

        self.spin_scale = QtWidgets.QSpinBox()
        self.spin_scale.valueChanged.connect(self.updateSizeLabel)
        self.spin_scale.setValue(int(600.0 / size.width()))
        self.spin_scale.setRange(1, 100)

        self.le_dpi = QtWidgets.QLineEdit("96")
        self.le_dpi.setValidator(QtGui.QIntValidator(1, 1000))

        self.combo_scalebar = QtWidgets.QComboBox()
        self.combo_scalebar.addItems(list(PngOptionsBox.item_positions.keys()))
        self.combo_scalebar.setCurrentText("Top Right")

        self.combo_label = QtWidgets.QComboBox()
        self.combo_label.addItems(list(PngOptionsBox.item_positions.keys()))
        self.combo_label.setCurrentText("Top Left")

        self.check_colorbar = QtWidgets.QCheckBox("Show colorbar.")
        self.check_colorbar.setChecked(True)

        self.check_raw = QtWidgets.QCheckBox("Export raw image data.")
        self.check_raw.toggled.connect(self.rawStateChanged)

        layout_scale = QtWidgets.QHBoxLayout()
        layout_scale.addWidget(self.spin_scale, 2)
        layout_scale.addWidget(self.label_x, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        layout_scale.addWidget(QtWidgets.QLabel("x"), 0)
        layout_scale.addWidget(self.label_y, 1)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Scale", layout_scale)
        layout.addRow("DPI", self.le_dpi)
        layout.addRow("Scalebar", self.combo_scalebar)
        layout.addRow("Element Label", self.combo_label)
        layout.addRow(self.check_colorbar)
        layout.addRow(self.check_raw)
        self.setLayout(layout)

    def updateSizeLabel(self) -> None:
        self.label_x.setText(f"{self.base_size.width() * self.spin_scale.value()}")
        self.label_y.setText(f"{self.base_size.height() * self.spin_scale.value()}")

    def imageSize(self) -> QtCore.QSize:
        return QtCore.QSize(
            self.base_size.width() * self.spin_scale.value(),
            self.base_size.height() * self.spin_scale.value(),
        )

    def scale(self) -> int:
        return self.spin_scale.value()

    def dpi(self) -> int:
        return int(self.le_dpi.text())

    def rawStateChanged(self, state: QtCore.Qt.CheckState) -> None:
        enabled = not self.check_raw.isChecked()
        self.spin_scale.setEnabled(enabled)
        self.le_dpi.setEnabled(enabled)
        self.combo_scalebar.setEnabled(enabled)
        self.combo_label.setEnabled(enabled)
        self.check_colorbar.setEnabled(enabled)

    def labelAlignment(self) -> QtCore.Qt.AlignmentFlag | None:
        return PngOptionsBox.item_positions[self.combo_label.currentText()]

    def scalebarAlignment(self) -> QtCore.Qt.AlignmentFlag | None:
        return PngOptionsBox.item_positions[self.combo_scalebar.currentText()]

    def useColorbar(self) -> bool:
        return self.check_colorbar.isChecked()

    def isRaw(self) -> bool:
        return self.check_raw.isChecked()

    def isComplete(self) -> bool:
        return self.le_dpi.hasAcceptableInput()


class RBGOptionsBox(OptionsBox):
    item_positions = {
        "Top Left": QtCore.Qt.AlignmentFlag.AlignTop
        | QtCore.Qt.AlignmentFlag.AlignLeft,
        "Top Right": QtCore.Qt.AlignmentFlag.AlignTop
        | QtCore.Qt.AlignmentFlag.AlignRight,
        "Bottom Left": QtCore.Qt.AlignmentFlag.AlignBottom
        | QtCore.Qt.AlignmentFlag.AlignLeft,
        "Bottom Right": QtCore.Qt.AlignmentFlag.AlignBottom
        | QtCore.Qt.AlignmentFlag.AlignRight,
        "Off": None,
    }

    def __init__(
        self,
        size: QtCore.QSize,
        elements: list[RGBLaserImageItem.RGBElement],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(
            "PNG Images",
            ".png",
            visible=True,
            allow_export_all=False,
            allow_calibrate=False,
        )

        self.rgb_elements = elements

        self.base_size = size
        self.label_x = QtWidgets.QLabel()
        self.label_y = QtWidgets.QLabel()

        self.spin_scale = QtWidgets.QSpinBox()
        self.spin_scale.valueChanged.connect(self.updateSizeLabel)
        self.spin_scale.setValue(int(600.0 / size.width()))
        self.spin_scale.setRange(1, 100)

        self.le_dpi = QtWidgets.QLineEdit("96")
        self.le_dpi.setValidator(QtGui.QIntValidator(1, 1000))

        self.combo_scalebar = QtWidgets.QComboBox()
        self.combo_scalebar.addItems(list(PngOptionsBox.item_positions.keys()))
        self.combo_scalebar.setCurrentText("Top Right")

        self.combo_label = QtWidgets.QComboBox()
        self.combo_label.addItems(list(PngOptionsBox.item_positions.keys()))
        self.combo_label.setCurrentText("Top Left")

        self.combo_venn = QtWidgets.QComboBox()
        self.combo_venn.addItems(list(PngOptionsBox.item_positions.keys()))
        self.combo_venn.setCurrentText("Bottom Left")

        self.check_raw = QtWidgets.QCheckBox("Export raw image data.")
        self.check_raw.toggled.connect(self.rawStateChanged)

        layout_scale = QtWidgets.QHBoxLayout()
        layout_scale.addWidget(self.spin_scale, 2)
        layout_scale.addWidget(self.label_x, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        layout_scale.addWidget(QtWidgets.QLabel("x"), 0)
        layout_scale.addWidget(self.label_y, 1)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Scale", layout_scale)
        layout.addRow("DPI", self.le_dpi)
        layout.addRow("Scalebar", self.combo_scalebar)
        layout.addRow("Element Label", self.combo_label)
        layout.addRow("Color Venn", self.combo_venn)
        layout.addRow(self.check_raw)
        self.setLayout(layout)

    def elements(self) -> list[str]:
        return [rgb.element for rgb in self.rgb_elements]

    def colors(self) -> list[QtGui.QColor]:
        return [rgb.color for rgb in self.rgb_elements]

    def ranges(self) -> list[tuple[float, float]]:
        return [rgb.prange for rgb in self.rgb_elements]

    def updateSizeLabel(self) -> None:
        self.label_x.setText(f"{self.base_size.width() * self.spin_scale.value()}")
        self.label_y.setText(f"{self.base_size.height() * self.spin_scale.value()}")

    def scale(self) -> int:
        return self.spin_scale.value()

    def imageSize(self) -> QtCore.QSize:
        return QtCore.QSize(
            self.base_size.width() * self.spin_scale.value(),
            self.base_size.height() * self.spin_scale.value(),
        )

    def dpi(self) -> int:
        return int(self.le_dpi.text())

    def rawStateChanged(self, state: QtCore.Qt.CheckState) -> None:
        enabled = not self.check_raw.isChecked()
        self.spin_scale.setEnabled(enabled)
        self.le_dpi.setEnabled(enabled)
        self.combo_scalebar.setEnabled(enabled)
        self.combo_label.setEnabled(enabled)
        self.combo_venn.setEnabled(enabled)

    def labelAlignment(self) -> QtCore.Qt.AlignmentFlag | None:
        return PngOptionsBox.item_positions[self.combo_label.currentText()]

    def scalebarAlignment(self) -> QtCore.Qt.AlignmentFlag | None:
        return PngOptionsBox.item_positions[self.combo_scalebar.currentText()]

    def vennAlignment(self) -> QtCore.Qt.AlignmentFlag | None:
        return PngOptionsBox.item_positions[self.combo_venn.currentText()]

    def isRaw(self) -> bool:
        return self.check_raw.isChecked()

    def isComplete(self) -> bool:
        return self.le_dpi.hasAcceptableInput()


class VtiOptionsBox(OptionsBox):
    def __init__(
        self,
        spacing: tuple[float, float, float],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(
            "VTK Images", ".vti", visible=True, allow_export_all=False, parent=parent
        )
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

    def spacing(self) -> tuple[float, float, float]:
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
        self,
        options: list[OptionsBox] | None = None,
        parent: QtWidgets.QWidget | None = None,
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

    def __init__(
        self, options: list[OptionsBox], parent: QtWidgets.QWidget | None = None
    ):
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

        filename_regexp = QtCore.QRegularExpression(f"[^{self.invalid_chars}]+")
        self.lineedit_filename.setValidator(
            QtGui.QRegularExpressionValidator(filename_regexp)
        )
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

    def getPath(self, name: str | None = None) -> Path:
        if name is None:
            name = self.lineedit_filename.text()
        path = Path(name).with_suffix(self.options.currentExt())
        return Path(self.lineedit_directory.text()).joinpath(path)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 200)

    def isComplete(self) -> bool:
        if not Path(self.lineedit_directory.text()).exists():
            return False
        # if self.options.indexForExt(Path(self.lineedit_filename.text()).suffix) == -1:
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
    def __init__(
        self,
        item: LaserImageItem,
        parent: QtWidgets.QWidget | None = None,
    ):
        aspect = item.pixelSize().width() / item.pixelSize().height()
        apparent_size = item.imageSize()
        if aspect > 1:
            apparent_size.setWidth(int(apparent_size.width() * aspect))
        elif aspect < 1:
            apparent_size.setHeight(int(apparent_size.height() / aspect))

        if isinstance(item, RGBLaserImageItem):
            options = [RBGOptionsBox(apparent_size, item.current_elements)]
        else:
            spacing = (
                item.laser.config.get_pixel_width(),
                item.laser.config.get_pixel_height(),
                item.laser.config.spotsize / 2.0,
            )
            options = [
                OptionsBox(
                    "Numpy Archives",
                    ".npz",
                    allow_calibrate=False,
                    allow_export_all=False,
                ),
                OptionsBox("CSV Document", ".csv"),
                PngOptionsBox(apparent_size),
                VtiOptionsBox(spacing),
            ]
        super().__init__(options, parent)

        self.item = item

        self.check_calibrate = QtWidgets.QCheckBox("Calibrate data.")
        self.check_calibrate.setChecked(True)
        self.check_calibrate.setToolTip("Calibrate the data before exporting.")

        self.check_export_all = QtWidgets.QCheckBox("Export all elements.")
        self.check_export_all.setToolTip(
            "Export all elements for the current image.\n"
            "The filename will be appended with the elements name."
        )
        self.check_export_all.clicked.connect(self.updatePreview)

        self.layout.insertWidget(2, self.check_calibrate)
        self.layout.insertWidget(3, self.check_export_all)

        # A default path
        path = self.pathForLaser(self.item.laser).resolve()
        self.lineedit_directory.setText(str(path.parent))
        self.lineedit_filename.setText(str(path.name).translate(self.invalid_map))
        self.typeChanged(0)

    def pathForLaser(self, laser: Laser, ext: str = ".npz") -> Path:
        path = Path(laser.info.get("File Path", ""))
        return path.with_name(laser.info.get("Name", "laser") + ext)

    def allowCalibrate(self) -> bool:
        return self.options.currentOption().allowCalibrate()

    def allowExportAll(self) -> bool:
        return self.options.currentOption().allowExportAll()

    def isCalibrate(self) -> bool:
        return self.check_calibrate.isChecked() and self.check_calibrate.isEnabled()

    def isExportAll(self) -> bool:
        return self.check_export_all.isChecked() and self.check_export_all.isEnabled()

    def isComplete(self) -> bool:
        suffix = Path(self.lineedit_filename.text()).suffix
        if suffix != "" and self.options.indexForExt(suffix) == -1:
            return False
        return super().isComplete()

    def updatePreview(self) -> None:
        path = Path(self.lineedit_filename.text())
        if self.isExportAll():
            path = path.with_name(path.stem + "_<element>")

        if path.suffix == "":
            path = path.with_suffix(self.options.currentExt())
        self.lineedit_preview.setText(str(path))

    def typeChanged(self, index: int) -> None:
        super().typeChanged(index)
        # Enable or disable checks
        self.check_calibrate.setEnabled(self.allowCalibrate())
        self.check_export_all.setEnabled(self.allowExportAll())
        self.updatePreview()

    def getPathForElement(self, path: Path, element: str) -> Path:
        return path.with_name(
            path.stem + "_" + element.translate(self.invalid_map) + path.suffix
        )

    # def getPathForLayer(self, path: Path, layer: int) -> Path:
    #     return path.with_name(path.stem + "_layer" + str(layer) + path.suffix)

    def generatePaths(self, item: LaserImageItem) -> list[tuple[Path, str]]:
        paths: list[tuple[Path, str]] = [(self.getPath(), item.element())]
        if self.isExportAll():
            paths = [
                (self.getPathForElement(p, i), i)
                for i in item.laser.elements
                for (p, _) in paths
            ]

        return [p for p in paths if p[0] != ""]

    def export(
        self,
        path: Path,
        laser: Laser,
        element: str | None = None,
        graphics_options: GraphicsOptions | None = None,
    ) -> None:
        option = self.options.currentOption()

        if option.ext == ".csv":
            if element is not None and element in laser.elements:
                data = laser.get(element, self.isCalibrate(), flat=True)
                io.textimage.save(path, data)

        elif option.ext == ".png":
            assert graphics_options is not None
            if isinstance(option, RBGOptionsBox):
                size = option.imageSize()
                scale = option.scale()
                if any(x in laser.elements for x in option.elements()):
                    image = generate_rgb_laser_image(
                        laser,
                        option.elements(),
                        option.colors(),
                        option.ranges(),
                        graphics_options,
                        label_alignment=option.labelAlignment(),
                        scalebar_alignment=option.scalebarAlignment(),
                        venn_alignment=option.vennAlignment(),
                        raw=option.isRaw(),
                        size=size,
                        scale=scale,
                        dpi=option.dpi(),
                    )
                    image.setDotsPerMeterX(option.dpi() * 39.37007874)
                    image.setDotsPerMeterY(option.dpi() * 39.37007874)
                    image.save(str(path.absolute()))
            else:
                size = option.imageSize()
                scale = option.scale()
                if element is not None and element in laser.elements:
                    image = generate_laser_image(
                        laser,
                        element,
                        graphics_options,
                        label_alignment=option.labelAlignment(),
                        scalebar_alignment=option.scalebarAlignment(),
                        colorbar=option.useColorbar(),
                        calibrate=self.isCalibrate(),
                        raw=option.isRaw(),
                        size=size,
                        scale=scale,
                        dpi=option.dpi(),
                    )
                    image.setDotsPerMeterX(option.dpi() * 39.37007874)
                    image.setDotsPerMeterY(option.dpi() * 39.37007874)
                    image.save(str(path.absolute()))
        elif option.ext == ".vti":
            spacing = option.spacing()
            # Last axis (z) is negative for layer order
            spacing = spacing[0], spacing[1], -spacing[2]
            data = laser.get(calibrate=self.isCalibrate())
            io.vtk.save(path, data, spacing)

        elif option.ext == ".npz":  # npz
            io.npz.save(path, laser)

        else:
            raise ValueError(f"Unable to export file as '{option.ext}'.")

        logger.info(f"Exported {laser.info['Name']} to {path.name}.")

    def accept(self) -> None:
        paths = self.generatePaths(self.item)
        prompt = OverwriteFilePrompt()
        paths = [p for p in paths if prompt.promptOverwrite(str(p[0].resolve()))]

        if len(paths) == 0:
            return

        try:
            for path, element in paths:
                self.export(path, self.item.laser, element, self.item.options)
        except Exception as e:  # pragma: no cover
            logger.exception(e)
            QtWidgets.QMessageBox.critical(self, "Unable to Export!", str(e))
            return

        super().accept()


class ExportAllDialog(ExportDialog):
    def __init__(
        self, items: list[LaserImageItem], parent: QtWidgets.QWidget | None = None
    ):
        unique: Set[str] = set()
        for item in items:
            unique.update(item.laser.elements)
        elements = sorted(unique)

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.addItems(elements)

        self.lineedit_prefix = QtWidgets.QLineEdit("")
        self.lineedit_prefix.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(f"[^{self.invalid_chars}]+")
            )
        )
        self.lineedit_prefix.textChanged.connect(self.updatePreview)

        super().__init__(items[0], parent)
        self.setWindowTitle("Export All")
        self.items = items

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

    def getPath(self, name: str | None = None) -> Path:
        path = super().getPath(name)
        prefix = self.lineedit_prefix.text()
        if prefix != "":
            path = path.with_name(prefix + "_" + path.name)
        return path

    def generatePaths(self, item: LaserImageItem) -> list[tuple[Path, str]]:
        paths: list[tuple[Path, str]] = [
            (self.getPath(item.laser.info["Name"]), self.combo_element.currentText())
        ]
        if self.isExportAll():
            paths = [
                (self.getPathForElement(p, i), i)
                for i in item.laser.elements
                for (p, _) in paths
            ]

        return [p for p in paths if p[0] != ""]

    def accept(self) -> None:
        all_paths = []
        prompt = OverwriteFilePrompt()

        for item in self.items:
            paths = self.generatePaths(item)
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
            for paths, item in zip(all_paths, self.items):
                for path, element in paths:
                    dlg.setValue(exported)
                    if dlg.wasCanceled():
                        break
                    self.export(path, item.laser, element, item.options)
                    exported += 1
        except Exception as e:  # pragma: no cover
            logger.exception(e)
            QtWidgets.QMessageBox.critical(self, "Unable to Export!", str(e))
            dlg.cancel()
            return

        dlg.setValue(count)
        QtWidgets.QDialog.accept(self)
