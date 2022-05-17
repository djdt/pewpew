import copy
from io import BytesIO

import numpy as np
from pathlib import Path
import logging

from PySide2 import QtCore, QtGui, QtWidgets

from pewlib.laser import Laser
from pewlib.srr import SRRConfig
from pewlib.calibration import Calibration
from pewlib.config import Config

from pewpew.actions import qAction
from pewpew.events import DragDropRedirectFilter

from pewpew.graphics.imageitems import LaserImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions

from pewpew.threads import ImportThread

from pewpew.widgets import dialogs, exportdialogs
from pewpew.widgets.tools.calculator import CalculatorTool
from pewpew.widgets.views import TabView, TabViewWidget

from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


class LaserTabView(TabView):
    """Tabbed view for displaying laser images."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.config = Config()
        self.options = GraphicsOptions()

    def addLaser(self, laser: Laser) -> "LaserTabWidget":
        """Open image of a laser in a new tab."""
        if len(self.widgets()) > 0:
            widget = self.widgets()[0]
            widget.addLaser(laser)
        else:
            widget = LaserTabWidget(self.options, self)
            widget.addLaser(laser)
            name = laser.info.get("Name", "<No Name>")
            self.addTab(name, widget)

        return widget

    # Events
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_open)
        menu.popup(event.globalPos())
        event.accept()

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
        thread = ImportThread(paths, config=self.config, parent=self)

        progress.canceled.connect(thread.requestInterruption)
        thread.importStarted.connect(progress.setLabelText)
        thread.progressChanged.connect(progress.setValue)

        thread.importFinished.connect(self.addLaser)
        thread.importFailed.connect(logger.exception)
        thread.finished.connect(progress.close)

        thread.start()

    def openTool(self, tool: str, item: LaserImageItem) -> None:
        if tool == "Calculator":
            widget = CalculatorTool(item, view=self)
            pass
        else:
            raise ValueError(f"Invalid tool type {tool}.")
        pass
        self.addTab(f"Tool: {tool}", widget)

    def applyCalibration(self, calibration: Dict[str, Calibration]) -> None:
        """Set calibrations in all tabs."""
        for widget in self.widgets():
            if isinstance(widget, LaserTabWidget):
                widget.applyCalibration(calibration)

    def applyConfig(self, config: Config) -> None:
        """Set laser configurations in all tabs."""
        self.config = config
        for widget in self.widgets():
            if isinstance(widget, LaserTabWidget):
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

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
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


class LaserTabWidget(TabViewWidget):
    """Class that stores and displays a laser image.

    Tracks modification of the data, config, calibration and information.
    Create via `:func:pewpew.laser.LaserView.addLaser`.

    Args:
        laser: input
        options: graphics options for this widget
        view: parent view
    """

    def __init__(self, options: GraphicsOptions, view: Optional[LaserTabView] = None):
        super().__init__(view)

        self.graphics = LaserGraphicsView(options, parent=self)
        self.graphics.setMouseTracking(True)

        # self.graphics.cursorValueChanged.connect(self.updateCursorStatus)
        # self.graphics.colorbar.editRequested.connect(self.)

        self.graphics.scene().focusItemChanged.connect(self.focusItemChanged)
        self.graphics.scene().setStickyFocus(True)

        self.combo_element = LaserComboBox()
        self.combo_element.namesSelected.connect(self.updateNames)
        self.combo_element.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        self.action_copy_image = qAction(
            "insert-image",
            "Copy Scene &Image",
            "Copy scene to clipboard.",
            self.actionCopyImage,
        )
        # self.action_open = qAction("document-open", "Open")
        self.action_export_all = qAction(
            "document-save-as",
            "E&xport All",
            "Export all lasers.",
            self.dialogExportAll,
        )

        # === Toolbar actions ===
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
            "view-filter",
            "Selection Dialog",
            "Start the selection dialog.",
            lambda: self.openDialog("Selection", None),
        )

        self.action_ruler = qAction(
            "tool-measure",
            "Measure",
            "Use a ruler to measure distance.",
            self.graphics.startRulerWidget,
        )
        self.action_slice = qAction(
            "view-object-histogram-linear",
            "1D Slice",
            "Select and display a 1-dimensional slice of the image.",
            self.graphics.startSliceWidget,
        )

        self.action_zoom_out = qAction(
            "zoom-fit-best",
            "Reset Zoom",
            "Reset zoom to full image extent.",
            self.graphics.zoomReset,
        )

        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.addActions(
            [
                self.action_select_rect,
                self.action_select_lasso,
                self.action_select_dialog,
            ]
        )
        self.toolbar.addSeparator()
        self.toolbar.addActions([self.action_ruler, self.action_slice])
        self.toolbar.addSeparator()
        self.toolbar.addActions([self.action_zoom_out])

        self.graphics.viewport().installEventFilter(DragDropRedirectFilter(self))
        # Filters for setting active view
        self.graphics.viewport().installEventFilter(self)
        self.combo_element.installEventFilter(self)
        self.toolbar.installEventFilter(self)

        layout_bar = QtWidgets.QHBoxLayout()
        layout_bar.addWidget(self.toolbar)
        layout_bar.addStretch(1)
        # layout_bar.addWidget(self.combo_layers, 0, QtCore.Qt.AlignRight)
        layout_bar.addWidget(self.combo_element, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.graphics, 1)
        layout.addLayout(layout_bar)
        self.setLayout(layout)

    def addLaser(self, laser: Laser) -> None:
        item = LaserImageItem(laser, self.graphics.options)
        self.graphics.scene().addItem(item)

        # Connect dialog requests
        item.requestDialog.connect(self.openDialog)

        item.requestExport.connect(self.dialogExport)
        item.requestSave.connect(self.dialogSave)

        item.hoveredValueChanged.connect(self.updateCursorStatus)
        item.hoveredValueCleared.connect(self.clearCursorStatus)

        # Modification
        item.colortableChanged.connect(self.laserColortableChanged)
        item.modified.connect(lambda: self.setWindowModified(True))

        item.redraw()
        item.setFocus(QtCore.Qt.NoFocusReason)

    def laserItems(self) -> List[LaserImageItem]:
        return [
            item
            for item in self.graphics.scene().items()
            if isinstance(item, LaserImageItem)
        ]

    def uniqueElements(self) -> List[str]:
        elements = set([])
        for item in self.laserItems():
            elements.update(item.laser.elements)
        return list(elements)

    def laserColortableChanged(
        self, table: List[int], vmin: float, vmax: float, unit: str
    ) -> None:
        # Todo calculate this based on all open lasers?
        self.graphics.colorbar.updateTable(table, vmin, vmax, unit)
        self.graphics.invalidateScene()

    def focusItemChanged(
        self,
        new: QtWidgets.QGraphicsItem,
        old: QtWidgets.QGraphicsItem,
        reason: QtCore.Qt.FocusReason,
    ) -> None:
        if not isinstance(new, LaserImageItem) or old == new:
            return

        # Update the combo box
        try:  # Remove any existing connects to the element combo box
            self.combo_element.currentTextChanged.disconnect()
        except RuntimeError:
            pass

        self.combo_element.blockSignals(True)
        self.combo_element.clear()

        self.combo_element.addItems(new.laser.elements)
        self.combo_element.setCurrentText(new.element())
        self.combo_element.currentTextChanged.connect(
            lambda s: [new.setElement(s), new.redraw()]
        )

        self.combo_element.blockSignals(False)
        self.laserColortableChanged(
            new.image.colorTable(),
            new.vmin,
            new.vmax,
            new.laser.calibration[new.element()].unit,
        )

    # Virtual
    def refresh(self) -> None:
        """Redraw images."""
        for item in self.graphics.items():
            if isinstance(item, LaserImageItem):
                item.redraw()

        self.graphics.zoomReset()
        self.graphics.invalidateScene()
        super().refresh()

    # Other
    def clearCursorStatus(self) -> None:
        """Clear window statusbar, if it exists."""
        status_bar = self.view.window().statusBar()
        if status_bar is not None:
            status_bar.clearMessage()

    def updateCursorStatus(
        self, pos: QtCore.QPointF, data_pos: QtCore.QPoint, v: float
    ) -> None:
        """Updates the windows statusbar if it exists."""
        status_bar = self.view.window().statusBar()
        if status_bar is None:  # pragma: no cover
            return

        if self.graphics.options.units == "index":  # convert to indices
            x, y = data_pos.x(), data_pos.y()
        else:
            x, y = pos.x(), pos.y()

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

        self.setWindowModified(True)

    # Callbacks
    def openDialog(
        self,
        dialog: str,
        item: Optional[LaserImageItem] = None,
        selection: bool = False,
    ) -> QtWidgets.QDialog:
        if item is None:
            item = self.graphics.scene().focusItem()

        if dialog == "Calibration":
            dlg = dialogs.CalibrationDialog(
                item.laser.calibration, item.element(), parent=self
            )
            dlg.calibrationSelected.connect(item.applyCalibration)
            dlg.calibrationApplyAll.connect(self.view.applyCalibration)
        elif dialog == "Colocalisation":
            dlg = dialogs.ColocalisationDialog(
                item.laser.get(flat=True), item.mask if selection else None, parent=self
            )
        elif dialog == "Config":
            dlg = dialogs.ConfigDialog(item.laser.config, parent=self)
            dlg.configSelected.connect(item.applyConfig)
            dlg.configApplyAll.connect(self.view.applyConfig)
        elif dialog == "Information":
            dlg = dialogs.InformationDialog(item.laser.info, parent=self)
            dlg.infoChanged.connect(item.applyInformation)
        elif dialog == "Selection":
            # item = self.laserItems()[0]

            dlg = dialogs.SelectionDialog(item, parent=self)
            dlg.maskSelected.connect(item.select)
            self.refreshed.connect(dlg.refresh)
        elif dialog == "Statistics":
            dlg = dialogs.StatsDialog(
                item.laser.get(calibrate=item.options.calibrate, flat=True),
                item.mask if selection else np.ones(item.laser.shape, dtype=bool),
                {k: v.unit for k, v in item.laser.calibration.items()}
                if item.options.calibrate
                else {},
                item.element(),
                pixel_size=(
                    item.laser.config.get_pixel_width(),
                    item.laser.config.get_pixel_height(),
                ),
                parent=self,
            )

        else:
            raise ValueError(f"Dialog type {dialog} is invalid!")

        dlg.open()
        return dlg

    def dialogExport(self, item: Optional[LaserImageItem] = None) -> QtWidgets.QDialog:
        if item is None:
            item = self.graphics.scene().focusItem()
            assert item is not None

        dlg = exportdialogs.ExportDialog(item, parent=self)
        dlg.open()
        return dlg

    def dialogExportAll(self) -> QtWidgets.QDialog:
        dlg = exportdialogs.ExportAllDialog(self.laserItems(), parent=self)
        dlg.open()
        return dlg

    def dialogSave(
        self, item: Optional[LaserImageItem] = None
    ) -> Optional[QtWidgets.QDialog]:
        """Save the document to an '.npz' file.

        If not already associated with an '.npz' path a dialog is opened to select one.
        """
        if item is None:
            item = self.graphics.scene().focusItem()

        path = Path(item.laser.info["File Path"])
        if path.suffix.lower() == ".npz" and path.exists():
            item.saveToFile(path)
            # Todo window modifed - 1
            self.setWindowModified(False)
            return None
        else:
            path = Path(item.laser.info.get("File Path", ""))
            path = path.with_name(item.name() + ".npz")

        dlg = QtWidgets.QFileDialog(
            self, "Save File", str(path.resolve()), "Numpy archive(*.npz);;All files(*)"
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.fileSelected.connect(item.saveToFile)
        dlg.fileSelected.connect(lambda _: self.setWindowModified(False))
        dlg.open()
        return dlg

    def actionCopyImage(self) -> None:
        self.graphics.copyToClipboard()

    def actionDuplicate(self) -> None:
        """Duplicate document to a new tab."""
        self.view.addLaser(copy.deepcopy(self.laser))

    # Events
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        menu = QtWidgets.QMenu(self)
        # menu.addAction(self.action_duplicate)
        menu.addAction(self.action_copy_image)
        menu.addSeparator()

        if self.graphics.items():  # More than one laser
            # @Todo add export all
            pass
        menu.popup(event.globalPos())
        event.accept()

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
