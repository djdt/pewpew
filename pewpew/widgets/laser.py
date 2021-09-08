import copy
from io import BytesIO
import numpy as np
from pathlib import Path
import logging

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib import io
from pewlib.laser import Laser
from pewlib.srr import SRRLaser, SRRConfig
from pewlib.calibration import Calibration
from pewlib.config import Config

from pewpew.actions import qAction, qToolButton
from pewpew.events import DragDropRedirectFilter

from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions

from pewpew.threads import ImportThread

from pewpew.widgets import dialogs, exportdialogs
from pewpew.widgets.views import View, ViewSpace, _ViewWidget

from typing import Dict, List, Optional, Set, Union


logger = logging.getLogger(__name__)


class LaserViewSpace(ViewSpace):
    """A viewspace for displaying LaserWidget.

    Parameters:
        config: the default config
        options: image drawing options
    """

    def __init__(
        self,
        orientaion: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(orientaion, parent)
        self.config = Config()
        self.options = GraphicsOptions()

    def uniqueElements(self) -> List[str]:
        """Return set of unqiue elements from all open images."""
        elements: Set[str] = set()
        for view in self.views:
            for widget in view.widgets():
                if isinstance(widget, LaserWidget):
                    elements.update(widget.laser.elements)
        return sorted(elements)

    def createView(self) -> "LaserView":
        """Override for ViewSpace."""
        view = LaserView(self)
        view.numTabsChanged.connect(self.numTabsChanged)
        self.views.append(view)
        self.numViewsChanged.emit()
        return view

    def currentElement(self) -> Optional[str]:
        """The active element name from the active image."""
        widget = self.activeWidget()
        if widget is None:
            return None
        return widget.current_element

    def setCurrentElement(self, element: str) -> None:
        """Set the active element name in all views."""
        for view in self.views:
            view.setCurrentElement(element)

    def applyCalibration(self, calibration: Dict[str, Calibration]) -> None:
        """Set the calibration in all views."""
        for view in self.views:
            view.applyCalibration(calibration)

    def applyConfig(self, config: Config) -> None:
        """Set the laser configuration in all views."""
        self.config = copy.copy(config)
        for view in self.views:
            view.applyConfig(self.config)

    # Editing options
    def colortableRangeDialog(self) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.ColocalisationDialog` and apply result."""

        def applyDialog(dialog: dialogs.ApplyDialog) -> None:
            self.options._colorranges = dialog.ranges
            self.options.colorrange_default = dialog.default_range
            self.refresh()

        dlg = dialogs.ColorRangeDialog(
            self.options._colorranges,
            self.options.colorrange_default,
            self.uniqueElements(),
            current_element=self.currentElement(),
            parent=self,
        )
        dlg.combo_element.currentTextChanged.connect(self.setCurrentElement)
        dlg.applyPressed.connect(applyDialog)
        dlg.open()
        return dlg

    def configDialog(self) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.ConfigDialog` and apply result."""
        dlg = dialogs.ConfigDialog(self.config, parent=self)
        dlg.check_all.setChecked(True)
        dlg.check_all.setEnabled(False)
        dlg.configApplyAll.connect(self.applyConfig)
        dlg.open()
        return dlg

    def fontsizeDialog(self) -> QtWidgets.QDialog:
        """Simple dialog for editing image font size."""
        dlg = QtWidgets.QInputDialog(self)
        dlg.setWindowTitle("Fontsize")
        dlg.setLabelText("Fontisze:")
        dlg.setIntValue(self.options.font.pointSize())
        dlg.setIntRange(2, 96)
        dlg.setInputMode(QtWidgets.QInputDialog.IntInput)
        dlg.intValueSelected.connect(self.options.font.setPointSize)
        dlg.intValueSelected.connect(self.refresh)
        dlg.open()
        return dlg

    def setColortable(self, table: str) -> None:
        self.options.colortable = table
        self.refresh()

    def toggleCalibrate(self, checked: bool) -> None:
        self.options.calibrate = checked
        self.refresh()

    def toggleColorbar(self, checked: bool) -> None:
        self.options.items["colorbar"] = checked
        self.refresh()

    def toggleLabel(self, checked: bool) -> None:
        self.options.items["label"] = checked
        self.refresh()

    def toggleScalebar(self, checked: bool) -> None:
        self.options.items["scalebar"] = checked
        self.refresh()

    def toggleSmooth(self, checked: bool) -> None:
        self.options.smoothing = checked
        self.refresh()


class LaserView(View):
    """Tabbed view for displaying laser images."""

    def __init__(self, viewspace: LaserViewSpace):
        super().__init__(viewspace)
        self.action_open = qAction(
            "document-open", "&Open", "Open new document(s).", self.actionOpen
        )

    def addLaser(self, laser: Laser) -> "LaserWidget":
        """Open image of  a laser in a new tab."""
        widget = LaserWidget(laser, self.viewspace.options, self)
        name = laser.info.get("Name", "<No Name>")
        self.addTab(name, widget)
        return widget

    def setCurrentElement(self, element: str) -> None:
        """Set displayed element in all tabs."""
        for widget in self.widgets():
            if element in widget.laser.elements:
                widget.current_element = element

    # Events
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        event.accept()
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
        """Open `paths` as new laser images."""
        paths = [Path(path) for path in paths]  # Ensure Path

        progress = QtWidgets.QProgressDialog(
            "Importing...", "Cancel", 0, 0, parent=self
        )
        progress.setWindowTitle("Importing...")
        progress.setMinimumDuration(2000)
        thread = ImportThread(paths, config=self.viewspace.config, parent=self)

        progress.canceled.connect(thread.requestInterruption)
        thread.importStarted.connect(progress.setLabelText)
        thread.progressChanged.connect(progress.setValue)

        thread.importFinished.connect(self.addLaser)
        thread.importFailed.connect(logger.exception)
        thread.finished.connect(progress.close)

        thread.start()

    def applyCalibration(self, calibration: Dict[str, Calibration]) -> None:
        """Set calibrations in all tabs."""
        for widget in self.widgets():
            if isinstance(widget, LaserWidget):
                widget.applyCalibration(calibration)

    def applyConfig(self, config: Config) -> None:
        """Set laser configurations in all tabs."""
        for widget in self.widgets():
            if isinstance(widget, LaserWidget):
                widget.applyConfig(config)

    # Actions
    def actionOpen(self) -> QtWidgets.QDialog:
        """Opens a file dialog for loading new lasers."""
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
    """Combo box with a context menu for editing names."""

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
        event.accept()
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_edit_names)
        menu.popup(event.globalPos())


class LaserWidget(_ViewWidget):
    """Class that stores and displays a laser image.

    Tracks modification of the data, config, calibration and information.
    Create via `:func:pewpew.laser.LaserView.addLaser`.

    Args:
        laser: input
        options: graphics options for this widget
        view: parent view
    """

    def __init__(self, laser: Laser, options: GraphicsOptions, view: LaserView = None):
        super().__init__(view)
        self.laser = laser
        self.is_srr = isinstance(laser, SRRLaser)

        self.graphics = LaserGraphicsView(options, parent=self)
        self.graphics.cursorValueChanged.connect(self.updateCursorStatus)
        self.graphics.label.editRequested.connect(self.labelEditDialog)
        self.graphics.colorbar.editRequested.connect(
            self.viewspace.colortableRangeDialog
        )
        self.graphics.setMouseTracking(True)

        self.combo_layers = QtWidgets.QComboBox()
        self.combo_layers.addItem("*")
        self.combo_layers.addItems([str(i) for i in range(0, self.laser.layers)])
        self.combo_layers.currentIndexChanged.connect(self.refresh)
        if not self.is_srr:
            self.combo_layers.setEnabled(False)
            self.combo_layers.setVisible(False)

        self.combo_element = LaserComboBox()
        self.combo_element.namesSelected.connect(self.updateNames)
        self.combo_element.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.combo_element.currentIndexChanged.connect(self.refresh)
        self.populateElements()

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
        self.action_information = qAction(
            "documentinfo",
            "In&formation",
            "View and edit image information.",
            self.actionInformation,
        )
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
        self.action_widget_none = qAction(
            "transform-move",
            "Clear Widgets",
            "Closes any open widgets.",
            self.graphics.endWidget,
        )
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
        self.widgets_button.addAction(self.action_widget_none)
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

        self.graphics.viewport().installEventFilter(DragDropRedirectFilter(self))
        self.combo_element.installEventFilter(self)
        self.selection_button.installEventFilter(self)
        self.view_button.installEventFilter(self)

        layout_bar = QtWidgets.QHBoxLayout()
        layout_bar.addWidget(self.selection_button, 0, QtCore.Qt.AlignLeft)
        layout_bar.addWidget(self.widgets_button, 0, QtCore.Qt.AlignLeft)
        layout_bar.addWidget(self.view_button, 0, QtCore.Qt.AlignLeft)
        layout_bar.addStretch(1)
        layout_bar.addWidget(self.combo_layers, 0, QtCore.Qt.AlignRight)
        layout_bar.addWidget(self.combo_element, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.graphics, 1)
        layout.addLayout(layout_bar)
        self.setLayout(layout)

    @property
    def current_element(self) -> str:
        return self.combo_element.currentText()

    @current_element.setter
    def current_element(self, element: str) -> None:
        self.combo_element.setCurrentText(element)

    @property
    def current_layer(self) -> Optional[int]:
        if not self.is_srr or self.combo_layers.currentIndex() == 0:
            return None
        return int(self.combo_layers.currentText())

    # Virtual
    def refresh(self) -> None:
        """Redraw image."""
        self.graphics.drawLaser(
            self.laser, self.current_element, layer=self.current_layer
        )
        if self.graphics.widget is not None:
            self.graphics.widget.imageChanged(self.graphics.image, self.graphics.data)
        self.graphics.invalidateScene()
        super().refresh()

    def rename(self, text: str) -> None:
        """Set the 'Name' value of laser information."""
        self.laser.info["Name"] = text
        self.modified = True

    # Other
    def labelEditDialog(self, name: str) -> QtWidgets.QInputDialog:
        """Simple dialog for editing the label (and element name)."""
        dlg = QtWidgets.QInputDialog(self)
        dlg.setWindowTitle("Edit Name")
        dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
        dlg.setTextValue(name)
        dlg.setLabelText("Rename:")
        dlg.textValueSelected.connect(
            lambda s: self.renameElement(self.current_element, s)
        )
        dlg.open()
        return dlg

    def renameElement(self, old: str, new: str) -> None:
        """Rename a single element."""
        self.laser.rename({old: new})
        self.modified = True
        self.populateElements()
        self.current_element = new
        self.refresh()

    def laserName(self) -> str:
        return self.laser.info.get("Name", "<No Name>")

    def laserFilePath(self, ext: str = ".npz") -> Path:
        path = Path(self.laser.info.get("File Path", ""))
        return path.with_name(self.laserName() + ext)

    def populateElements(self) -> None:
        """Repopulate the element combo box."""
        self.combo_element.blockSignals(True)
        self.combo_element.clear()
        self.combo_element.addItems(self.laser.elements)
        self.combo_element.blockSignals(False)

    def clearCursorStatus(self) -> None:
        """Clear window statusbar, if it exists."""
        status_bar = self.viewspace.window().statusBar()
        if status_bar is not None:
            status_bar.clearMessage()

    def updateCursorStatus(self, x: float, y: float, v: float) -> None:
        """Updates the windows statusbar if it exists."""
        status_bar = self.viewspace.window().statusBar()
        if status_bar is None:  # pragma: no cover
            return

        if self.viewspace.options.units == "index":  # convert to indices
            p = self.graphics.mapToData(QtCore.QPointF(x, y))
            x, y = p.x(), p.y()

        if v is None:
            status_bar.clearMessage()
        elif np.isfinite(v):
            status_bar.showMessage(f"{x:.4g},{y:.4g} [{v:.4g}]")
        else:
            status_bar.showMessage(f"{x:.4g},{y:.4g} [nan]")

    def updateNames(self, rename: dict) -> None:
        """Rename multiple elements."""
        current = self.current_element
        self.laser.rename(rename)
        self.populateElements()
        current = rename[current]
        self.current_element = current

        self.modified = True

    # Transformations
    def cropToSelection(self) -> None:
        """Crop image to current selection and open in a new tab.

        If selection is not rectangular then it is filled with nan.
        """
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

        info = self.laser.info.copy()
        info["Name"] = self.laserName() + "_cropped"
        info["File Path"] = str(Path(info.get("File Path", "")).with_stem(info["Name"]))
        new_widget = self.view.addLaser(
            Laser(
                new_data,
                calibration=self.laser.calibration,
                config=self.laser.config,
                info=info,
            )
        )
        new_widget.setActive()

    def transform(self, flip: str = None, rotate: str = None) -> None:
        """Transform the image.

        Args:
            flip: flip the image ['horizontal', 'vertical']
            rotate: rotate the image 90 degrees ['left', 'right']

        """
        if self.is_srr:  # pragma: no cover
            QtWidgets.QMessageBox.information(
                self, "Transform", "Unable to transform SRR data."
            )
            return
        if flip is not None:
            if flip in ["horizontal", "vertical"]:
                axis = 1 if flip == "horizontal" else 0
                self.laser.data = np.flip(self.laser.data, axis=axis)
            else:
                raise ValueError("flip must be 'horizontal', 'vertical'.")
        if rotate is not None:
            if rotate in ["left", "right"]:
                k = 1 if rotate == "right" else 3 if rotate == "left" else 2
                self.laser.data = np.rot90(self.laser.data, k=k, axes=(1, 0))
            else:
                raise ValueError("rotate must be 'left', 'right'.")
        self.modified = True
        self.refresh()

    # Callbacks
    def applyCalibration(self, calibrations: Dict[str, Calibration]) -> None:
        """Set laser calibrations."""
        modified = False
        for element in calibrations:
            if element in self.laser.calibration:
                self.laser.calibration[element] = copy.copy(calibrations[element])
                modified = True
        if modified:
            self.modified = True
            self.refresh()

    def applyConfig(self, config: Config) -> None:
        """Set laser configuration."""
        # Only apply if the type of config is correct
        if type(config) is type(self.laser.config):  # noqa
            self.laser.config = copy.copy(config)
            self.modified = True
            self.refresh()

    def applyInformation(self, info: Dict[str, str]) -> None:
        """Set laser information."""
        # if self.laser.info["Name"] != info["Name"]:  # pragma: ignore
        #     self.view.tabs.setTabText(self.index(), info["Name"])
        if self.laser.info != info:
            self.laser.info = info
            self.modified = True

    def saveDocument(self, path: Union[str, Path]) -> None:
        """Saves the laser to an '.npz' file.

        See Also:
            `:func:pewlib.io.npz.save`
        """
        if isinstance(path, str):
            path = Path(path)

        io.npz.save(path, self.laser)
        self.laser.info["File Path"] = str(path.resolve())
        self.modified = False

    def actionCalibration(self) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.CalibrationDialog` and applies result."""
        dlg = dialogs.CalibrationDialog(
            self.laser.calibration, self.current_element, parent=self
        )
        dlg.calibrationSelected.connect(self.applyCalibration)
        dlg.calibrationApplyAll.connect(self.viewspace.applyCalibration)
        dlg.open()
        return dlg

    def actionConfig(self) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.ConfigDialog` and applies result."""
        dlg = dialogs.ConfigDialog(self.laser.config, parent=self)
        dlg.configSelected.connect(self.applyConfig)
        dlg.configApplyAll.connect(self.viewspace.applyConfig)
        dlg.open()
        return dlg

    def actionCopyImage(self) -> None:
        self.graphics.copyToClipboard()

    def actionCopySelectionText(self) -> None:
        """Copies the currently selected data to the system clipboard."""
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
        """Duplicate document to a new tab."""
        self.view.addLaser(copy.deepcopy(self.laser))

    def actionExport(self) -> QtWidgets.QDialog:
        """Opens a `:class:pewpew.exportdialogs.ExportDialog`.

        This can save the document to various formats.
        """
        dlg = exportdialogs.ExportDialog(self, parent=self)
        dlg.open()
        return dlg

    def actionInformation(self) -> QtWidgets.QDialog:
        """Opens a `:class:pewpew.widgets.dialogs.InformationDialog`."""
        dlg = dialogs.InformationDialog(self.laser.info, parent=self)
        dlg.infoChanged.connect(self.applyInformation)
        dlg.open()
        return dlg

    def actionSave(self) -> QtWidgets.QDialog:
        """Save the document to an '.npz' file.

        If not already associated with an '.npz' path a dialog is opened to select one.
        """
        path = Path(self.laser.info["File Path"])
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
        """Open a `:class:pewpew.widgets.dialogs.SelectionDialog` and applies selection."""
        dlg = dialogs.SelectionDialog(self.graphics, parent=self)
        dlg.maskSelected.connect(self.graphics.drawSelectionImage)
        self.refreshed.connect(dlg.refresh)
        dlg.show()
        return dlg

    def actionStatistics(self, crop_to_selection: bool = False) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.StatsDialog` with image data.

        Args:
            crop_to_selection: pass current selection as a mask
        """
        data = self.laser.get(calibrate=self.viewspace.options.calibrate, flat=True)
        mask = self.graphics.mask
        if mask is None or not crop_to_selection:
            mask = np.ones(data.shape, dtype=bool)

        units = {}
        if self.viewspace.options.calibrate:
            units = {k: v.unit for k, v in self.laser.calibration.items()}

        dlg = dialogs.StatsDialog(
            data,
            mask,
            units,
            self.current_element,
            pixel_size=(
                self.laser.config.get_pixel_width(),
                self.laser.config.get_pixel_height(),
            ),
            parent=self,
        )
        dlg.open()
        return dlg

    def actionStatisticsSelection(self) -> QtWidgets.QDialog:
        return self.actionStatistics(True)

    def actionColocal(self, crop_to_selection: bool = False) -> QtWidgets.QDialog:
        """Open a `:class:pewpew.widgets.dialogs.ColocalisationDialog` with image data.

        Args:
            crop_to_selection: pass current selection as a mask
        """
        data = self.laser.get(flat=True)
        mask = self.graphics.mask if crop_to_selection else None

        dlg = dialogs.ColocalisationDialog(data, mask, parent=self)
        dlg.open()
        return dlg

    def actionColocalSelection(self) -> QtWidgets.QDialog:
        return self.actionColocal(True)

    # Events
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        event.accept()
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
            menu.addAction(self.action_information)
            menu.addSeparator()
            menu.addAction(self.action_statistics)
            menu.addAction(self.action_colocalisation)
        menu.popup(event.globalPos())

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.Cancel):
            self.graphics.endSelection()
            self.graphics.endWidget()
        elif event.matches(QtGui.QKeySequence.Paste):
            mime = QtWidgets.QApplication.clipboard().mimeData()
            if mime.hasFormat("arplication/x-pew2config"):
                with BytesIO(mime.data("application/x-pew2config")) as fp:
                    array = np.load(fp)
                if self.is_srr:
                    config = SRRConfig.from_array(array)
                else:
                    config = Config.from_array(array)
                self.applyConfig(config)
            elif mime.hasFormat("application/x-pew2calibration"):
                with BytesIO(mime.data("application/x-pew2calibration")) as fp:
                    npy = np.load(fp)
                    calibrations = {k: Calibration.from_array(npy[k]) for k in npy}
                self.applyCalibration(calibrations)
        super().keyPressEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        self.refresh()
        super().showEvent(event)
