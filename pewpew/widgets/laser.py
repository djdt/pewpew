import copy
import numpy as np
import os
import logging

from PySide2 import QtCore, QtGui, QtWidgets

from pew import io
from pew.laser import Laser
from pew.srr import SRRLaser, SRRConfig
from pew.config import Config

from pewpew.lib.io import import_any, PEW_VALID_EXTS
from pewpew.lib.viewoptions import ViewOptions

from pewpew.actions import qAction, qToolButton
from pewpew.widgets.canvases import InteractiveLaserCanvas
from pewpew.widgets import dialogs, exportdialogs
from pewpew.widgets.views import View, ViewSpace, _ViewWidget

from typing import List, Set, Tuple


logger = logging.getLogger(__name__)


class LaserViewSpace(ViewSpace):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.config = Config()
        self.options = ViewOptions()

    def uniqueIsotopes(self) -> List[str]:
        isotopes: Set[str] = set()
        for view in self.views:
            for widget in view.widgets():
                if isinstance(widget, LaserWidget):
                    isotopes.update(widget.laser.isotopes)
        return sorted(isotopes)

    def createView(self) -> "LaserView":
        view = LaserView(self)
        view.numTabsChanged.connect(self.numTabsChanged)
        self.views.append(view)
        self.numViewsChanged.emit()
        return view

    def currentIsotope(self) -> str:
        widget = self.activeWidget()
        if widget is None:
            return None
        return widget.combo_isotope.currentText()

    def setCurrentIsotope(self, isotope: str) -> None:
        for view in self.views:
            view.setCurrentIsotope(isotope)

    def applyCalibration(self, calibration: dict) -> None:
        for view in self.views:
            view.applyCalibration(calibration)

    def applyConfig(self, config: Config) -> None:
        self.config = copy.copy(config)
        for view in self.views:
            view.applyConfig(self.config)


class LaserView(View):
    def __init__(self, viewspace: LaserViewSpace):
        super().__init__(viewspace)
        self.action_open = qAction(
            "document-open", "&Open", "Open new document(s).", self.actionOpen
        )

    def addLaser(self, laser: Laser) -> "LaserWidget":
        widget = LaserWidget(laser, self.viewspace.options, self)
        name = laser.name if laser.name != "" else os.path.basename(laser.path)
        self.addTab(name, widget)
        return widget

    def setCurrentIsotope(self, isotope: str) -> None:
        for widget in self.widgets():
            if isotope in widget.laser.isotopes:
                widget.combo_isotope.setCurrentText(isotope)

    # Events
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_open)
        menu.popup(event.globalPos())

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            return super().dropEvent(event)

        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if any(url.fileName().lower().endswith(ext) for ext in PEW_VALID_EXTS)
        ]
        try:
            lasers = import_any(paths, self.viewspace.config)
            for laser in lasers:
                self.addLaser(laser)
            event.acceptProposedAction()
        except io.error.PewException as e:  # pragma: no cover
            event.ignore()
            logger.exception(e)
            QtWidgets.QMessageBox.critical(self, type(e).__name__, f"{e}")

    # Callbacks
    def openDocument(self, paths: List[str]) -> None:
        try:
            for laser in import_any(paths, self.viewspace.config):
                self.addLaser(laser)

        except io.error.PewException as e:  # pragma: no cover
            logger.exception(e)
            QtWidgets.QMessageBox.critical(self, type(e).__name__, f"{e}")

    def applyCalibration(self, calibration: dict) -> None:
        for widget in self.widgets():
            if isinstance(widget, LaserWidget):
                widget.applyCalibration(calibration)

    def applyConfig(self, config: Config) -> None:
        for widget in self.widgets():
            if isinstance(widget, LaserWidget):
                widget.applyConfig(config)

    # Actions
    def actionOpen(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QFileDialog(
            self,
            "Open File(s).",
            "",
            "CSV Documents(*.csv *.txt *.text);;Numpy Archives(*.npz);;All files(*)",
        )
        dlg.selectNameFilter("All files(*)")
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dlg.filesSelected.connect(self.openDocument)
        dlg.open()
        return dlg


class LaserWidget(_ViewWidget):
    def __init__(self, laser: Laser, viewoptions: ViewOptions, view: LaserView = None):
        super().__init__(view)
        self.laser = laser
        self.is_srr = isinstance(laser, SRRLaser)

        self.canvas = InteractiveLaserCanvas(viewoptions, parent=self)
        self.canvas.cursorClear.connect(self.clearCursorStatus)
        self.canvas.cursorMoved.connect(self.updateCursorStatus)
        # We have our own ConnectionRefusedErrorxt menu so hide the normal one
        self.canvas.setContextMenuPolicy(QtCore.Qt.NoContextMenu)

        self.combo_layers = QtWidgets.QComboBox()
        self.combo_layers.addItem("*")
        self.combo_layers.addItems([str(i) for i in range(0, self.laser.layers)])
        self.combo_layers.currentIndexChanged.connect(self.refresh)
        if not self.is_srr:
            self.combo_layers.setEnabled(False)
            self.combo_layers.setVisible(False)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_isotope.currentIndexChanged.connect(self.refresh)
        self.populateIsotopes()

        self.action_select_none = qAction(
            "transform-move",
            "Clear Selection",
            "Clear any selections.",
            self.canvas.clearSelection,
        )
        self.action_select_rect = qAction(
            "draw-rectangle",
            "Rectangle Selector",
            "Start the rectangle selector tool.",
            self.canvas.startRectangleSelection,
        )
        self.action_select_lasso = qAction(
            "draw-freehand",
            "Lasso Selector",
            "Start the lasso selector tool.",
            self.canvas.startLassoSelection,
        )
        self.action_select_dialog = qAction(
            "dialog-information",
            "Selection Dialog",
            "Start the selection dialog.",
            self.actionSelectDialog,
        )
        self.selection_button = qToolButton("select", "Selection")
        self.selection_button.addAction(self.action_select_none)
        self.selection_button.addAction(self.action_select_rect)
        self.selection_button.addAction(self.action_select_lasso)
        self.selection_button.addAction(self.action_select_dialog)

        self.action_ruler = qAction(
            "tool-measure",
            "Measure",
            "Use a ruler to measure distance.",
            self.canvas.startRuler,
        )
        self.widgets_button = qToolButton("tool-measure", "Widgets")
        self.widgets_button.addAction(self.action_ruler)

        self.action_zoom_in = qAction(
            "zoom-in",
            "Zoom to Area",
            "Start zoom area selection.",
            self.canvas.startZoom,
        )
        self.action_zoom_out = qAction(
            "zoom-original",
            "Reset Zoom",
            "Reset zoom to full imgae extent.",
            self.canvas.unzoom,
        )
        self.view_button = qToolButton("zoom", "Zoom")
        self.view_button.addAction(self.action_zoom_in)
        self.view_button.addAction(self.action_zoom_out)

        self.canvas.installEventFilter(self)
        self.combo_isotope.installEventFilter(self)
        self.selection_button.installEventFilter(self)
        self.view_button.installEventFilter(self)

        layout_bar = QtWidgets.QHBoxLayout()
        layout_bar.addWidget(self.selection_button, 0, QtCore.Qt.AlignLeft)
        layout_bar.addWidget(self.widgets_button, 0, QtCore.Qt.AlignLeft)
        layout_bar.addWidget(self.view_button, 0, QtCore.Qt.AlignLeft)
        layout_bar.addStretch(1)
        layout_bar.addWidget(self.combo_layers, 0, QtCore.Qt.AlignRight)
        layout_bar.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        layout.addLayout(layout_bar)
        self.setLayout(layout)

        self.createActions()

    # Virtual
    def refresh(self) -> None:
        if not self.is_srr or self.combo_layers.currentIndex() == 0:
            layer = None
        else:
            layer = int(self.combo_layers.currentText())

        self.canvas.endSelection()
        self.canvas.drawLaser(self.laser, self.combo_isotope.currentText(), layer=layer)

    def rename(self, text: str) -> None:
        self.laser.name = text
        self.setModified(True)

    @QtCore.Slot("QWidget*")
    def mouseSelectStart(self, callback_widget: QtWidgets.QWidget) -> None:
        self.canvas.installEventFilter(callback_widget)

    @QtCore.Slot("QWidget*")
    def mouseSelectEnd(self, callback_widget: QtWidgets.QWidget) -> None:
        self.canvas.removeEventFilter(callback_widget)

    # Other
    def laserFilePath(self, ext: str = ".npz") -> str:
        return os.path.join(os.path.dirname(self.laser.path), self.laser.name + ext)

    def populateIsotopes(self) -> None:
        self.combo_isotope.blockSignals(True)
        self.combo_isotope.clear()
        self.combo_isotope.addItems(self.laser.isotopes)
        self.combo_isotope.blockSignals(False)

    def clearCursorStatus(self) -> None:
        status_bar = self.viewspace.window().statusBar()
        if status_bar is not None:
            status_bar.clearMessage()

    def updateCursorStatus(self, x: float, y: float, v: float) -> None:
        status_bar = self.viewspace.window().statusBar()
        if status_bar is None:
            return
        unit = self.canvas.viewoptions.units

        if not self.is_srr or self.combo_layers.currentIndex() == 0:
            px = self.laser.config.get_pixel_width()
            py = self.laser.config.get_pixel_height()
        else:
            layer = int(self.combo_layers.currentText())
            px = self.laser.config.get_pixel_width(layer)
            py = self.laser.config.get_pixel_height(layer)

        if unit == "row":
            x, y = self.laser.shape[0] - int(y / py) - 1, int(x / px)
        elif unit == "second":
            x = x / self.laser.config.speed
            y = 0
        if np.isfinite(v):
            status_bar.showMessage(f"{x:.4g},{y:.4g} [{v:.4g}]")
        else:
            status_bar.showMessage(f"{x:.4g},{y:.4g} [nan]")

    # Transformations
    def crop(self, new_extent: Tuple[float, float, float, float] = None) -> None:
        if self.is_srr:
            QtWidgets.QMessageBox.information(
                self, "Transform", "Unable to transform SRR data."
            )
            return
        if new_extent is None:
            # Default is to crop to current view limits.
            new_extent = self.canvas.view_limits
        if new_extent == self.canvas.extent:
            # Extent is same
            return
        extent = self.canvas.extent
        w, h = extent[1] - extent[0], extent[3] - extent[2]
        sy, sx = self.laser.data.shape
        data = self.laser.data[
            sy - int(new_extent[3] / h * sy) : sy - int(new_extent[2] / h * sy),
            int(new_extent[0] / w * sx) : int(new_extent[1] / w * sx),
        ]

        new_widget = self.view.addLaser(
            Laser(
                data,
                calibration=self.laser.calibration,
                config=self.laser.config,
                name=self.laser.name + "_cropped",
            )
        )
        new_widget.setActive()

    def transform(self, flip: str = None, rotate: str = None) -> None:
        if self.is_srr:  # pragma: no cover
            QtWidgets.QMessageBox.information(
                self, "Transform", "Unable to transform SRR data."
            )
            return
        if flip is not None:
            axis = 1 if flip == "horizontal" else 0
            self.laser.data = np.flip(self.laser.data, axis=axis)
        if rotate is not None:
            k = 1 if rotate == "right" else 3 if rotate == "left" else 2
            self.laser.data = np.rot90(self.laser.data, k=k, axes=(1, 0))
        self.setModified(True)
        self.refresh()

    # Callbacks
    def applyCalibration(self, calibrations: dict) -> None:
        modified = False
        for isotope in calibrations:
            if isotope in self.laser.calibration:
                self.laser.calibration[isotope] = copy.copy(calibrations[isotope])
                modified = True
        if modified:
            self.setModified(True)
            self.refresh()

    def applyConfig(self, config: Config) -> None:
        # Only apply if the type of config is correct
        if isinstance(config, SRRConfig) == self.is_srr:
            self.laser.config = copy.copy(config)
            self.setModified(True)
            self.refresh()

    def saveDocument(self, path: str) -> None:
        io.npz.save(path, [self.laser])
        self.laser.path = path
        self.setModified(False)

    # Actions
    def createActions(self) -> None:
        self.action_calibration = qAction(
            "go-top",
            "Ca&libration",
            "Edit the documents calibration.",
            self.actionCalibration,
        )
        self.action_config = qAction(
            "document-edit", "&Config", "Edit the document's config.", self.actionConfig
        )
        self.action_copy_image = qAction(
            "insert-image",
            "Copy &Image",
            "Copy image to clipboard.",
            self.actionCopyImage,
        )
        self.action_duplicate = qAction(
            "edit-copy",
            "Duplicate image",
            "Open a copy of the image.",
            self.actionDuplicate,
        )
        self.action_export = qAction(
            "document-save-as", "E&xport", "Export documents.", self.actionExport
        )
        self.action_export.setShortcut("Ctrl+X")
        # Add the export action so we can use it via shortcut
        self.addAction(self.action_export)
        self.action_save = qAction(
            "document-save", "&Save", "Save document to numpy archive.", self.actionSave
        )
        self.action_save.setShortcut("Ctrl+S")
        # Add the save action so we can use it via shortcut
        self.addAction(self.action_save)
        self.action_statistics = qAction(
            "dialog-information",
            "Statistics",
            "Open statisitics dialog for selected data.",
            self.actionStatistics,
        )
        self.action_colocalisation = qAction(
            "dialog-information",
            "Colocalisation",
            "Open the colocalisation dialog.",
            self.actionColocal,
        )

    def actionCalibration(self) -> QtWidgets.QDialog:
        dlg = dialogs.CalibrationDialog(
            self.laser.calibration, self.combo_isotope.currentText(), parent=self
        )
        dlg.calibrationSelected.connect(self.applyCalibration)
        dlg.calibrationApplyAll.connect(self.viewspace.applyCalibration)
        dlg.open()
        return dlg

    def actionConfig(self) -> QtWidgets.QDialog:
        dlg = dialogs.ConfigDialog(self.laser.config, parent=self)
        dlg.configSelected.connect(self.applyConfig)
        dlg.configApplyAll.connect(self.viewspace.applyConfig)
        dlg.open()
        return dlg

    def actionCopyImage(self) -> None:
        self.canvas.copyToClipboard()

    def actionDuplicate(self) -> None:
        self.view.addLaser(copy.deepcopy(self.laser))

    def actionExport(self) -> QtWidgets.QDialog:
        dlg = exportdialogs.ExportDialog(self, parent=self)
        dlg.open()
        return dlg

    def actionSave(self) -> QtWidgets.QDialog:
        path = self.laser.path
        if path.lower().endswith(".npz") and os.path.exists(path):
            self.saveDocument(path)
            return None
        else:
            path = self.laserFilePath()
        dlg = QtWidgets.QFileDialog(
            self, "Save File", path, "Numpy archive(*.npz);;All files(*)"
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.fileSelected.connect(self.saveDocument)
        dlg.open()
        return dlg

    def actionSelectDialog(self) -> QtWidgets.QDialog:
        dlg = dialogs.SelectionDialog(
            self.laser.get(flat=True), self.combo_isotope.currentText(), parent=self
        )
        dlg.maskSelected.connect(self.canvas.setSelection)
        dlg.open()
        return dlg

    def actionStatistics(self) -> QtWidgets.QDialog:
        mask = self.canvas.getSelection()
        if mask is None:
            mask = np.full(self.laser.shape, True, dtype=bool)

        dlg = dialogs.StatsDialog(
            self.laser.get(flat=True),
            mask,
            self.combo_isotope.currentText(),
            pixel_size=(
                self.laser.config.get_pixel_width(),
                self.laser.config.get_pixel_height(),
            ),
            coloroptions=self.viewspace.options.colors,
            parent=self,
        )
        dlg.open()
        return dlg

    def actionColocal(self) -> QtWidgets.QDialog:
        mask = self.canvas.getSelection()
        dlg = dialogs.ColocalisationDialog(self.laser.get(flat=True), mask, parent=self)
        dlg.open()
        return dlg

    # Events
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_duplicate)
        menu.addAction(self.action_copy_image)
        menu.addSeparator()
        menu.addAction(self.view.action_open)
        menu.addAction(self.action_save)
        menu.addAction(self.action_export)
        menu.addSeparator()
        menu.addAction(self.action_config)
        menu.addAction(self.action_calibration)
        menu.addAction(self.action_statistics)
        menu.addAction(self.action_colocalisation)
        menu.popup(event.globalPos())

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        self.refresh()
        super().showEvent(event)
