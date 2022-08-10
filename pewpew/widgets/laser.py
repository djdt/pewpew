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

from pewpew.graphics.imageitems import ImageOverlayItem, LaserImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions

from pewpew.threads import ImportThread

from pewpew.widgets import dialogs, exportdialogs
from pewpew.widgets.controls import LaserControlBar
from pewpew.widgets.tools import ToolWidget
from pewpew.widgets.tools.calculator import CalculatorTool
from pewpew.widgets.tools.filtering import FilteringTool
from pewpew.widgets.tools.overlays import OverlayTool
from pewpew.widgets.tools.standards import StandardsTool
from pewpew.widgets.views import TabView, TabViewWidget

from typing import Dict, List, Optional, Union


logger = logging.getLogger(__name__)


# @todo, add button to toolbars to lock the current item as active


class LaserTabView(TabView):
    """Tabbed view for displaying laser images."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.config = Config()
        self.options = GraphicsOptions()

    # def addLaser(self, laser: Laser) -> "LaserTabWidget":
    #     """Open image of a laser in a new tab."""
    #     if len(self.widgets()) > 0:
    #         widget = self.widgets()[0]
    #         widget.addLaser(laser)
    #     else:
    #         widget = LaserTabWidget(self.options, self)
    #         widget.addLaser(laser)
    #         name = laser.info.get("Name", "<No Name>")
    #         self.addTab(name, widget)

    #     return widget

    def importFile(self, data: Union[Laser, QtGui.QImage]) -> "LaserTabWidget":
        if len(self.widgets()) > 0:
            widget = self.widgets()[0]
        else:
            widget = LaserTabWidget(self.options, self)
            self.addTab("1", widget)

        if isinstance(data, Laser):
            widget.addLaser(data)
        else:
            widget.addImage(data)

    # Events
    # def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
    #     menu = QtWidgets.QMenu(self)
    #     menu.addAction(self.action_open)
    #     menu.popup(event.globalPos())
    #     event.accept()

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

        thread.importFinished.connect(self.importFile)
        thread.importFailed.connect(logger.exception)
        thread.finished.connect(progress.close)

        thread.start()

    def applyCalibration(self, calibration: Dict[str, Calibration]) -> None:
        """Set calibrations in all tabs."""
        for widget in self.widgets():
            if isinstance(widget, LaserTabWidget):
                for item in widget.laserItems():
                    item.applyCalibration(calibration)

    def applyConfig(self, config: Config) -> None:
        """Set laser configurations in all tabs."""
        self.config = config
        for widget in self.widgets():
            if isinstance(widget, LaserTabWidget):
                for item in widget.laserItems():
                    item.applyConfig(config)

    # Actions
    # def actionOpenLaser(self) -> QtWidgets.QDialog:
    #     """Opens a file dialog for loading new lasers."""
    #     dlg = QtWidgets.QFileDialog(
    #         self,
    #         "Open File(s).",
    #         "",
    #         "CSV Documents(*.csv *.txt *.text);;Numpy Archives(*.npz);;All files(*)",
    #     )
    #     dlg.selectNameFilter("All files(*)")
    #     dlg.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
    #     dlg.filesSelected.connect(self.openDocument)
    #     dlg.open()
    #     return dlg


# class LaserComboBox(QtWidgets.QComboBox):
#     """Combo box with a context menu for editing names."""

#     namesSelected = QtCore.Signal(dict)

#     def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
#         super().__init__(parent)
#         self.action_edit_names = qAction(
#             "document-edit",
#             "Edit Names",
#             "Edit image names.",
#             self.actionNameEditDialog,
#         )

#     def actionNameEditDialog(self) -> QtWidgets.QDialog:
#         names = [self.itemText(i) for i in range(self.count())]
#         dlg = dialogs.NameEditDialog(names, parent=self)
#         dlg.namesSelected.connect(self.namesSelected)
#         dlg.open()
#         return dlg

#     def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
#         event.accept()
#         menu = QtWidgets.QMenu(self)
#         menu.addAction(self.action_edit_names)
#         menu.popup(event.globalPos())


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

        self.graphics.scene().focusItemChanged.connect(self.updateForItem)
        self.graphics.scene().setStickyFocus(True)

        self.action_copy_image = qAction(
            "insert-image",
            "Copy Scene &Image",
            "Copy scene to clipboard.",
            self.graphics.copyToClipboard,
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

        self.controls = QtWidgets.QStackedWidget()
        self.laser_controls = LaserControlBar()

        self.controls.addWidget(self.laser_controls)

        self.laser_controls.toolbar.addActions(
            [
                self.action_select_rect,
                self.action_select_lasso,
                self.action_select_dialog,
            ]
        )
        self.laser_controls.toolbar.addSeparator()
        self.laser_controls.toolbar.addActions([self.action_ruler, self.action_slice])
        self.laser_controls.toolbar.addSeparator()
        self.laser_controls.toolbar.addActions([self.action_zoom_out])

        self.graphics.viewport().installEventFilter(DragDropRedirectFilter(self))
        # Filters for setting active view
        self.graphics.viewport().installEventFilter(self)
        # self.laser_controls.elements.installEventFilter(self)
        self.controls.installEventFilter(self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.graphics, 1)
        layout.addWidget(self.controls, 0)
        self.setLayout(layout)

    def addLaser(self, laser: Laser) -> None:
        item = LaserImageItem(laser, self.graphics.options)

        # Connect dialog requests
        item.requestDialog.connect(self.openDialog)
        item.requestTool.connect(self.openTool)

        item.requestExport.connect(self.dialogExport)
        item.requestSave.connect(self.dialogSave)

        item.hoveredValueChanged.connect(self.updateCursorStatus)
        item.hoveredValueCleared.connect(self.clearCursorStatus)

        # Modification
        item.modified.connect(lambda: self.setWindowModified(True))

        item.redraw()
        item.setActive(True)
        self.updateForItem(item)
        self.graphics.scene().addItem(item)
        self.graphics.zoomReset()

    def addImage(self, path: Union[str, Path]) -> None:
        if isinstance(path, Path):
            path = str(path.absolute())
        image = QtGui.QImage(path)

        item = ImageOverlayItem(
            image, QtCore.QRectF(0, 0, image.width(), image.height())
        )
        item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
        item.setAcceptedMouseButtons(QtCore.Qt.RightButton)
        item.requestDialog.connect(self.openDialog)

        self.graphics.scene().addItem(item)
        self.graphics.zoomReset()

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

    def updateForItem(
        self,
        new: QtWidgets.QGraphicsItem,
        old: Optional[QtWidgets.QGraphicsItem] = None,
        reason: QtCore.Qt.FocusReason = QtCore.Qt.NoFocusReason,
    ) -> None:
        # Todo: test if lock active
        if old == new or new is None:
            return

        if isinstance(new, LaserImageItem):
            self.controls.setCurrentWidget(self.laser_controls)

            self.laser_controls.setItem(new)
        elif isinstance(new, ImageOverlayItem):  # Todo: maybe add a proper class?
            pass
        else:
            raise ValueError(f"updateForItem: Unknown item type {type(new)}.")

    # Virtual
    def refresh(self) -> None:
        """Redraw images."""
        for item in self.graphics.items():
            if isinstance(item, LaserImageItem):
                item.redraw()

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

    # def updateNames(self, rename: dict) -> None:
    #     """Rename multiple elements."""
    #     current = self.current_element
    #     self.laser.rename(rename)
    #     self.populateElements()
    #     current = rename[current]
    #     self.current_element = current

    #     self.setWindowModified(True)

    # Callbacks
    def openDialog(
        self,
        dialog: str,
        item: Optional[Union[ImageOverlayItem, LaserImageItem]] = None,
        selection: bool = False,
    ) -> QtWidgets.QDialog:
        if item is None:
            item = self.graphics.scene().focusItem()

        # Shared dialogs
        if dialog == "Pixel Size":
            dlg = dialogs.PixelSizeDialog(item.pixelSize(), parent=self)
            dlg.sizeSelected.connect(item.setPixelSize)
            print("pixelsize")
        elif not isinstance(item, LaserImageItem):
            raise ValueError(f"Item {item} is not a LaserImageItem, dialog {dialog} invalid.")
        # Laser dialogs
        elif dialog == "Calibration":
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

    def openTool(self, tool: str, item: LaserImageItem) -> ToolWidget:
        if tool == "Calculator":
            widget = CalculatorTool(item, view=self.view)
        elif tool == "Filtering":
            widget = FilteringTool(item, view=self.view)
        elif tool == "Overlay":
            widget = OverlayTool(item, view=self.view)
        elif tool == "Standards":
            widget = StandardsTool(item, view=self.view)
        else:
            raise ValueError(f"Invalid tool type {tool}.")

        widget.itemModified.connect(self.updateForItem)
        self.view.addTab(f"Tool: {tool}", widget)
        self.view.setActiveWidget(widget)
        return widget

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

    def actionDuplicate(self) -> None:
        """Duplicate document to a new tab."""
        self.view.addLaser(copy.deepcopy(self.laser))

    # Events
    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        if not self.graphics.underMouse():
            return
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
