import copy
import numpy as np
from pathlib import Path
import logging

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib import io
from pewlib.laser import Laser
from pewlib.srr import SRRLaser, SRRConfig
from pewlib.config import Config

from pewpew.actions import qAction, qToolButton

from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions

from pewpew.threads import ImportThread

from pewpew.widgets import dialogs, exportdialogs
from pewpew.widgets.views import View, ViewSpace, _ViewWidget

from typing import List, Set, Tuple, Union


logger = logging.getLogger(__name__)


class LaserViewSpace(ViewSpace):
    def __init__(
        self,
        orientaion: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(orientaion, parent)
        self.config = Config()
        self.options = GraphicsOptions()

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
        return widget.current_isotope

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
        name = laser.name if laser.name != "" else laser.path.stem
        self.addTab(name, widget)
        return widget

    def setCurrentIsotope(self, isotope: str) -> None:
        for widget in self.widgets():
            if isotope in widget.laser.isotopes:
                widget.current_isotope = isotope

    # Events
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_open)
        menu.popup(event.globalPos())

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:  # pragma: no cover
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if not event.mimeData().hasUrls():  # pragma: no cover
            return super().dropEvent(event)

        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        event.acceptProposedAction()

        self.openDocument(paths)

    # Callbacks
    def openDocument(self, paths: List[Path]) -> None:
        progress = QtWidgets.QProgressDialog(
            "Importing...", "Cancel", 0, 0, parent=self
        )
        progress.setMinimumDuration(2000)
        thread = ImportThread(paths, config=self.viewspace.config, parent=self)

        progress.canceled.connect(thread.requestInterruption)
        thread.importStarted.connect(progress.setLabelText)
        thread.progressChanged.connect(progress.setValue)

        thread.importFinished.connect(self.addLaser)
        thread.importFailed.connect(logger.exception)
        thread.finished.connect(progress.close)

        thread.start()

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


class LaserComboBox(QtWidgets.QComboBox):
    namesSelected = QtCore.Signal(dict)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.action_edit_names = qAction(
            "document-edit",
            "Edit Names",
            "Edit image names.",
            self.actionNameEditDialog,
        )

    def actionNameEditDialog(self) -> QtWidgets.QDialog:
        names = [self.itemText(i) for i in range(self.count())]
        dlg = dialogs.NameEditDialog(names, parent=self)
        dlg.namesSelected.connect(self.namesSelected)
        dlg.open()
        return dlg

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_edit_names)
        menu.popup(event.globalPos())


class LaserWidget(_ViewWidget):
    def __init__(self, laser: Laser, options: GraphicsOptions, view: LaserView = None):
        super().__init__(view)
        self.laser = laser
        self.is_srr = isinstance(laser, SRRLaser)

        self.graphics = LaserGraphicsView(options, parent=self)
        # We have our own ConnectionRefusedErrorxt menu so hide the normal one
        # self.canvas.cursorClear.connect(self.clearCursorStatus)
        # self.canvas.cursorMoved.connect(self.updateCursorStatus)

        self.combo_layers = QtWidgets.QComboBox()
        self.combo_layers.addItem("*")
        self.combo_layers.addItems([str(i) for i in range(0, self.laser.layers)])
        self.combo_layers.currentIndexChanged.connect(self.refresh)
        if not self.is_srr:
            self.combo_layers.setEnabled(False)
            self.combo_layers.setVisible(False)

        self.combo_isotope = LaserComboBox()
        self.combo_isotope.namesSelected.connect(self.updateNames)
        self.combo_isotope.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_isotope.currentIndexChanged.connect(self.refresh)
        self.populateIsotopes()

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
            "Open the statisitics dialog.",
            self.actionStatistics,
        )
        self.action_select_statistics = qAction(
            "dialog-information",
            "Selection Statistics",
            "Open the statisitics dialog for the current selection.",
            self.actionStatisticsSelection,
        )
        self.action_colocalisation = qAction(
            "dialog-information",
            "Colocalisation",
            "Open the colocalisation dialog.",
            self.actionColocal,
        )
        self.action_select_colocalisation = qAction(
            "dialog-information",
            "Selection Colocalisation",
            "Open the colocalisation dialog for the current selection.",
            self.actionColocalSelection,
        )

        # Toolbar actions
        self.action_select_none = qAction(
            "transform-move",
            "Clear Selection",
            "Clear any selections.",
            self.graphics.endSelection,
        )
        self.action_select_none.setShortcut("Escape")
        self.action_select_rect = qAction(
            "draw-rectangle",
            "Rectangle Selector",
            "Start the rectangle selector tool, use 'Shift' "
            "to add to selection and 'Control' to subtract.",
            self.graphics.startRectangleSelection,
        )
        self.action_select_lasso = qAction(
            "draw-freehand",
            "Lasso Selector",
            "Start the lasso selector tool, use 'Shift' "
            "to add to selection and 'Control' to subtract.",
            self.graphics.startLassoSelection,
        )
        self.action_select_dialog = qAction(
            "dialog-information",
            "Selection Dialog",
            "Start the selection dialog.",
            self.actionSelectDialog,
        )
        self.action_select_copy_text = qAction(
            "insert-table",
            "Copy Selection as Text",
            "Copy the current selection to the clipboard as a column of text values.",
            self.actionCopySelectionText,
        )
        self.action_select_crop = qAction(
            "transform-crop",
            "Crop to Selection",
            "Crop the image to the current selection.",
            self.actionCropSelection,
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
            self.graphics.startRulerWidget,
        )
        self.action_slice = qAction(
            "tool-measure",
            "1D Slice",
            "Select and display a 1-dimensional slice of the image.",
            self.graphics.startSliceWidget,
        )
        self.widgets_button = qToolButton("tool-measure", "Widgets")
        self.widgets_button.addAction(self.action_ruler)
        self.widgets_button.addAction(self.action_slice)

        self.action_zoom_in = qAction(
            "zoom-in",
            "Zoom to Area",
            "Start zoom area selection.",
            self.graphics.zoomStart,
        )
        self.action_zoom_out = qAction(
            "zoom-original",
            "Reset Zoom",
            "Reset zoom to full image extent.",
            self.graphics.zoomReset,
        )
        self.view_button = qToolButton("zoom", "Zoom")
        self.view_button.addAction(self.action_zoom_in)
        self.view_button.addAction(self.action_zoom_out)

        self.graphics.installEventFilter(self)
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
        layout.addWidget(self.graphics, 1)
        layout.addLayout(layout_bar)
        self.setLayout(layout)

    @property
    def current_isotope(self) -> str:
        return self.combo_isotope.currentText()

    @current_isotope.setter
    def current_isotope(self, isotope: str) -> None:
        self.combo_isotope.setCurrentText(isotope)

    @property
    def current_layer(self) -> int:
        if not self.is_srr or self.combo_layers.currentIndex() == 0:
            return None
        return int(self.combo_layers.currentText())

    # Virtual
    def refresh(self) -> None:
        self.graphics.widget = None
        self.graphics.drawLaser(
            self.laser, self.current_isotope, layer=self.current_layer
        )
        self.graphics.invalidateScene()
        super().refresh()

    def rename(self, text: str) -> None:
        self.laser.name = text
        self.modified = True

    # Other
    def laserFilePath(self, ext: str = ".npz") -> Path:
        return self.laser.path.parent.joinpath(self.laser.name + ext)

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
        if status_bar is None:  # pragma: no cover
            return
        unit = self.graphics.options.units

        layer = self.current_layer
        if layer is None:
            px = self.laser.config.get_pixel_width()
            py = self.laser.config.get_pixel_height()
        else:  # pragma: no cover
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

    def updateNames(self, rename: dict) -> None:
        current = self.current_isotope
        self.laser.rename(rename)
        self.populateIsotopes()
        current = rename[current]
        self.current_isotope = current

    # Transformations
    def crop(self, new_extent: Tuple[float, float, float, float] = None) -> None:
        if self.is_srr:  # pragma: no cover
            QtWidgets.QMessageBox.information(
                self, "Transform", "Unable to transform SRR data."
            )
            return
        if new_extent is None:  # pragma: no cover
            # Default is to crop to current view limits.
            new_extent = self.graphics.view_limits
        if new_extent == self.graphics.extent:  # pragma: no cover
            # Extent is same
            return
        extent = self.graphics.extent
        w, h = extent[1] - extent[0], extent[3] - extent[2]
        sy, sx = self.laser.data.shape
        data = self.laser.data[
            int(new_extent[2] / h * sy) : int(new_extent[3] / h * sy),
            int(new_extent[0] / w * sx) : int(new_extent[1] / w * sx),
        ]

        path = self.laser.path
        new_widget = self.view.addLaser(
            Laser(
                data.copy(),
                calibration=self.laser.calibration,
                config=self.laser.config,
                name=self.laser.name + "_cropped",
                path=path.parent.joinpath(self.laser.name + "_cropped" + path.suffix),
            )
        )
        new_widget.setActive()

    def cropToSelection(self) -> None:
        if self.is_srr:  # pragma: no cover
            QtWidgets.QMessageBox.information(
                self, "Transform", "Unable to transform SRR data."
            )
            return

        mask = self.graphics.mask
        if mask is None or np.all(mask == 0):  # pragma: no cover
            return
        ix, iy = np.nonzero(mask)
        x0, x1, y0, y1 = np.min(ix), np.max(ix) + 1, np.min(iy), np.max(iy) + 1

        data = self.laser.data
        new_data = np.empty((x1 - x0, y1 - y0), dtype=data.dtype)
        for name in new_data.dtype.names:
            new_data[name] = np.where(
                mask[x0:x1, y0:y1], data[name][x0:x1, y0:y1], np.nan
            )

        path = self.laser.path
        new_widget = self.view.addLaser(
            Laser(
                new_data,
                calibration=self.laser.calibration,
                config=self.laser.config,
                name=self.laser.name + "_cropped",
                path=path.parent.joinpath(self.laser.name + "_cropped" + path.suffix),
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
        self.modified = True
        self.refresh()

    # Callbacks
    def applyCalibration(self, calibrations: dict) -> None:
        modified = False
        for isotope in calibrations:
            if isotope in self.laser.calibration:
                self.laser.calibration[isotope] = copy.copy(calibrations[isotope])
                modified = True
        if modified:
            self.modified = True
            self.refresh()

    def applyConfig(self, config: Config) -> None:
        # Only apply if the type of config is correct
        if isinstance(config, SRRConfig) == self.is_srr:
            self.laser.config = copy.copy(config)
            self.modified = True
            self.refresh()

    def saveDocument(self, path: Union[str, Path]) -> None:
        if isinstance(path, str):
            path = Path(path)

        io.npz.save(path, self.laser)
        self.laser.path = path
        self.modified = False

    def actionCalibration(self) -> QtWidgets.QDialog:
        dlg = dialogs.CalibrationDialog(
            self.laser.calibration, self.current_isotope, parent=self
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
        self.graphics.copyToClipboard()

    def actionCopySelectionText(self) -> None:
        data = self.graphics.data[self.graphics.mask].ravel()

        html = (
            '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
            "<table>"
        )
        text = ""
        for x in data:
            html += f"<tr><td>{x:.10g}</td></tr>"
            text += f"{x:.10g}\n"
        html += "</table>"

        mime = QtCore.QMimeData()
        mime.setHtml(html)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def actionCropSelection(self) -> None:
        self.cropToSelection()

    def actionDuplicate(self) -> None:
        self.view.addLaser(copy.deepcopy(self.laser))

    def actionExport(self) -> QtWidgets.QDialog:
        dlg = exportdialogs.ExportDialog(self, parent=self)
        dlg.open()
        return dlg

    def actionSave(self) -> QtWidgets.QDialog:
        path = self.laser.path
        if path.suffix.lower() == ".npz" and path.exists():
            self.saveDocument(path)
            return None
        else:
            path = self.laserFilePath()
        dlg = QtWidgets.QFileDialog(
            self, "Save File", str(path.resolve()), "Numpy archive(*.npz);;All files(*)"
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.fileSelected.connect(self.saveDocument)
        dlg.open()
        return dlg

    def actionSelectDialog(self) -> QtWidgets.QDialog:
        dlg = dialogs.SelectionDialog(self.graphics, parent=self)
        dlg.maskSelected.connect(self.graphics.drawSelectionImage)
        self.refreshed.connect(dlg.refresh)
        dlg.show()
        return dlg

    def actionStatistics(self, crop_to_selection: bool = False) -> QtWidgets.QDialog:
        data = self.laser.get(calibrate=self.viewspace.options.calibrate, flat=True)
        if crop_to_selection:
            mask = self.graphics.mask
        else:
            mask = np.ones(data.shape, dtype=np.bool)

        units = {}
        if self.viewspace.options.calibrate:
            units = {k: v.unit for k, v in self.laser.calibration.items()}

        dlg = dialogs.StatsDialog(
            data,
            mask,
            units,
            self.current_isotope,
            pixel_size=(
                self.laser.config.get_pixel_width(),
                self.laser.config.get_pixel_height(),
            ),
            colorranges=None,
            parent=self,
        )
        dlg.open()
        return dlg

    def actionStatisticsSelection(self) -> QtWidgets.QDialog:
        return self.actionStatistics(True)

    def actionColocal(self, crop_to_selection: bool = False) -> QtWidgets.QDialog:
        data = self.laser.get(flat=True)
        mask = self.graphics.mask if crop_to_selection else None

        dlg = dialogs.ColocalisationDialog(data, mask, parent=self)
        dlg.open()
        return dlg

    def actionColocalSelection(self) -> QtWidgets.QDialog:
        return self.actionColocal(True)

    # Events
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QtWidgets.QMenu(self)
        # menu.addAction(self.action_duplicate)
        menu.addAction(self.action_copy_image)
        menu.addSeparator()

        if self.graphics.posInSelection(event.pos()):
            menu.addAction(self.action_select_copy_text)
            menu.addAction(self.action_select_crop)
            menu.addSeparator()
            menu.addAction(self.action_select_statistics)
            menu.addAction(self.action_select_colocalisation)
        else:
            menu.addAction(self.view.action_open)
            menu.addAction(self.action_save)
            menu.addAction(self.action_export)
            menu.addSeparator()
            menu.addAction(self.action_config)
            menu.addAction(self.action_calibration)
            menu.addSeparator()
            menu.addAction(self.action_statistics)
            menu.addAction(self.action_colocalisation)
        menu.popup(event.globalPos())

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        self.refresh()
        super().showEvent(event)
