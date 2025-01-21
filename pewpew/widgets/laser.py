import copy
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from pewlib import io
from pewlib.calibration import Calibration
from pewlib.config import Config, SpotConfig
from pewlib.laser import Laser
from pewlib.process.register import overlap_structured_arrays
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction
from pewpew.events import DragDropRedirectFilter
from pewpew.graphics.imageitems import (
    ImageOverlayItem,
    LaserImageItem,
    RGBLaserImageItem,
    SnapImageItem,
)
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions
from pewpew.threads import ImportThread
from pewpew.widgets import dialogs, exportdialogs
from pewpew.widgets.controls import (
    ControlBar,
    ImageControlBar,
    LaserControlBar,
    RGBLaserControlBar,
)
from pewpew.widgets.tools import ToolWidget
from pewpew.widgets.tools.calculator import CalculatorTool
from pewpew.widgets.tools.filtering import FilteringTool
from pewpew.widgets.tools.standards import StandardsTool
from pewpew.widgets.views import TabView, TabViewWidget

logger = logging.getLogger(__name__)


# @todo, add button to toolbars to lock the current item as active


class LaserTabView(TabView):
    """Tabbed view for displaying laser images."""

    fileImported = QtCore.Signal(Path)
    numImageItemsChanged = QtCore.Signal()  # needs close
    numLaserItemsChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.config = Config()
        self.options = GraphicsOptions()
        self.tabs.setAutoHide(True)

    def insertTab(self, index: int, text: str, widget: "LaserTabWidget") -> int:
        index = super().insertTab(index, text, widget)
        if isinstance(widget, LaserTabWidget):
            widget.numLaserItemsChanged.connect(self.numLaserItemsChanged)
            widget.numImageItemsChanged.connect(self.numImageItemsChanged)
        return index

    def newLaserTab(self) -> "LaserTabWidget":
        widget = LaserTabWidget(self.options, self)
        number = sum(1 for x in self.widgets() if isinstance(x, LaserTabWidget))
        self.addTab(f"Tab {number + 1}", widget)
        return widget

    def importFile(
        self, path: Path, data: Laser | tuple[Laser, QtCore.QPointF] | QtGui.QImage
    ) -> "LaserTabWidget":
        try:
            widget = self.activeWidget()
            if not isinstance(widget, LaserTabWidget):
                widget = next(
                    iter(w for w in self.widgets() if isinstance(w, LaserTabWidget))
                )
        except StopIteration:
            widget = self.newLaserTab()
        if isinstance(data, Laser):
            widget.addLaser(data)
        elif isinstance(data, tuple):
            assert isinstance(data[0], Laser)
            assert isinstance(data[1], QtCore.QPointF)
            widget.addLaser(data[0], pos=data[1])
        else:
            widget.addImage(data)

        self.fileImported.emit(path)
        return widget

    # Events
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
            # logs / imzml go to mainwindow for wizard
            if not any(
                io.laser.is_nwi_laser_log(path) or io.imzml.is_imzml(path)
                for path in paths
            ):
                event.acceptProposedAction()
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if not event.mimeData().hasUrls():  # pragma: no cover
            return super().dropEvent(event)

        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        event.acceptProposedAction()
        self.openDocument(paths)

    # Callbacks
    def openDocument(self, paths: list[Path] | Path) -> None:
        """Open `paths` as new laser images."""
        if isinstance(paths, (Path, str)):
            paths = [paths]

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

    def applyCalibration(self, calibration: dict[str, Calibration]) -> None:
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

    def laserItems(self) -> list[LaserImageItem]:
        items = []
        for widget in self.widgets():
            if isinstance(widget, LaserTabWidget):
                items.extend(widget.laserItems())
        return items

    def focusLaserItem(self) -> LaserImageItem | None:
        widget = self.activeWidget()
        if isinstance(widget, LaserTabWidget):
            item = widget.graphics.scene().focusItem()
            if isinstance(item, LaserImageItem):
                return item
        return None

    def uniqueElements(self) -> list[str]:
        elements = set([])
        for widget in self.widgets():
            if isinstance(widget, LaserTabWidget):
                elements.update(widget.uniqueElements())
        return list(elements)

    def setElement(self, element: str) -> None:
        for widget in self.widgets():
            if isinstance(widget, LaserTabWidget):
                widget.laser_controls.elements.setCurrentText(element)


class LaserTabWidget(TabViewWidget):
    numImageItemsChanged = QtCore.Signal()
    numLaserItemsChanged = QtCore.Signal()

    """Class that stores and displays a laser image.

    Tracks modification of the data, config, calibration and information.
    Create via `:func:pewpew.laser.LaserView.addLaser`.

    Args:
        laser: input
        options: graphics options for this widget
        view: parent view
    """

    def __init__(self, options: GraphicsOptions, view: LaserTabView | None = None):
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

        # === Alignment actions ===
        self.action_align_auto = qAction(
            "view-refresh",
            "FFT Register",
            "Register all images to the topmost image.",
            self.graphics.alignLaserItemsFFT,
        )
        self.action_align_horz = qAction(
            "align-vertical-top",
            "Align Left to Right",
            "Layout images in a horizontal line.",
            self.graphics.alignLaserItemsLeftToRight,
        )
        self.action_align_vert = qAction(
            "align-horizontal-left",
            "Align Top to Bottom",
            "Layout items in a vertical line.",
            self.graphics.alignLaserItemsTopToBottom,
        )

        self.action_merge = qAction(
            "merge",
            "Merge Images",
            "Merge all laser images into a single file",
            self.mergeLaserItems,
        )

        # === Toolbar selections actions ===
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
        # === Toolbar widget actions ===
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

        # === Toolbar transform actions ===
        self.action_transform_affine = qAction(
            "node-segment-line",
            "3-Point Affine",
            "Use an affine point tool for transformation, this will not affect the pixel size!",
            lambda: self.graphics.startTransformAffine(None),
        )
        self.action_transform_scale = qAction(
            "transform-rotate",
            "Scale and Rotate",
            "Scale and rotate image, this will not affect the pixel size!",
            lambda: self.graphics.startTransformScale(None),
        )
        self.action_transform_reset = qAction(
            "edit-reset",
            "Reset Transform",
            "Resets the image transformation.",
            lambda: self.graphics.resetTransform(None),
        )

        # === Toolbart view actions ===

        self.action_zoom_out = qAction(
            "zoom-fit-best",
            "Reset Zoom",
            "Reset zoom to full image extent.",
            self.graphics.zoomReset,
        )

        self.controls = QtWidgets.QStackedWidget()
        self.no_controls = ControlBar()
        self.laser_controls = LaserControlBar()
        self.rgb_laser_controls = RGBLaserControlBar()
        self.image_controls = ImageControlBar()

        self.controls.addWidget(self.no_controls)
        self.controls.addWidget(self.laser_controls)
        self.controls.addWidget(self.rgb_laser_controls)
        self.controls.addWidget(self.image_controls)

        for index in range(self.controls.count()):
            self.controls.widget(index).toolbar.addActions([self.action_zoom_out])
            self.controls.widget(index).toolbar.addSeparator()

        for control in [self.laser_controls, self.rgb_laser_controls]:
            control.toolbar.addActions(
                [
                    self.action_select_rect,
                    self.action_select_lasso,
                    self.action_select_dialog,
                ]
            )
            control.toolbar.addSeparator()
            control.toolbar.addActions([self.action_ruler, self.action_slice])

        self.image_controls.toolbar.addActions(
            [
                self.action_transform_affine,
                self.action_transform_scale,
                self.action_transform_reset,
            ]
        )
        self.image_controls.toolbar.addActions([self.action_ruler])

        self.graphics.viewport().installEventFilter(DragDropRedirectFilter(self))
        # Filters for setting active view
        self.graphics.viewport().installEventFilter(self)
        # self.laser_controls.elements.installEventFilter(self)
        self.controls.installEventFilter(self)

        self.view_toolbar = QtWidgets.QToolBar()
        self.view_toolbar.addActions([self.action_zoom_out])

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.graphics, 1)
        layout.addWidget(self.controls, 0)
        self.setLayout(layout)
        self.show()  # Call show here so that zoomReset works

    def addLaser(
        self, laser: Laser, pos: QtCore.QPointF | None = None
    ) -> "LaserImageItem":
        item = LaserImageItem(laser, self.graphics.options)
        self.addLaserItem(item, pos=pos)
        return item

    def addLaserItem(
        self, item: LaserImageItem, pos: QtCore.QPointF | None = None
    ) -> None:
        item.requestDialog.connect(self.openDialog)
        item.requestTool.connect(self.openTool)
        item.requestConversion.connect(self.convertImage)

        item.requestAddLaser.connect(self.addLaser)
        item.requestExport.connect(self.dialogExport)
        item.requestSave.connect(self.dialogSave)

        item.elementsChanged.connect(lambda: self.updateForItem(item))

        item.hoveredValueChanged.connect(self.updateCursorStatus)
        item.hoveredValueCleared.connect(self.clearCursorStatus)

        # Modification
        item.modified.connect(lambda: self.setWindowModified(True))
        item.destroyed.connect(self.numLaserItemsChanged)

        item.redraw()
        item.setActive(True)

        self.updateForItem(item)
        self.graphics.scene().addItem(item)
        if pos is not None:
            item.setPos(pos)
        else:
            self.graphics.zoomReset()

        self.numLaserItemsChanged.emit()

    def addImage(self, path: str | Path) -> "ImageOverlayItem":
        if isinstance(path, Path):
            path = str(path.absolute())
        image = QtGui.QImage(path)

        item = ImageOverlayItem(
            image, QtCore.QRectF(0, 0, image.width(), image.height()), path=path
        )
        item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
        item.requestDialog.connect(self.openDialog)
        item.destroyed.connect(self.numImageItemsChanged)

        self.updateForItem(item)
        self.graphics.scene().addItem(item)
        self.graphics.zoomReset()

        self.numImageItemsChanged.emit()
        return item

    def convertImage(self, to_type: str, item: SnapImageItem | None = None) -> None:
        if item is None:
            item = self.graphics.scene().focusItem()

        if to_type == "RGBLaserImageItem":
            if not isinstance(item, LaserImageItem):
                raise ValueError(f"cannot convert {type(item)} to a RGBLaserImageItem")
            new_item = RGBLaserImageItem.fromLaserImageItem(item, self.graphics.options)
        elif to_type == "LaserImageItem":
            if not isinstance(item, RGBLaserImageItem):
                raise ValueError(f"cannot convert {type(item)} to a LaserImageItem")
            new_item = LaserImageItem(item.laser, self.graphics.options)

        self.graphics.scene().removeItem(item)
        self.addLaserItem(new_item, item.pos())

    def laserItems(self) -> list[LaserImageItem]:
        return self.graphics.laserItems()

    def selectedLaserItems(self) -> list[LaserImageItem]:
        return self.graphics.selectedLaserItems()

    def mergeLaserItems(self) -> None:
        items = self.selectedLaserItems()[::-1]
        if len(items) == 0:
            items = self.laserItems()[::-1]
        if len(items) < 2:
            return

        if not all([item.pixelSize() == items[0].pixelSize() for item in items[1:]]):
            QtWidgets.QMessageBox.warning(
                self, "Unable to Merge", "All images must have the same pixel size."
            )
            return

        datas = [item.laser.get(calibrate=False) for item in items]
        positions = [item.mapToData(item.pos()) for item in items]
        offsets = [(p.y(), p.x()) for p in positions]
        merge = overlap_structured_arrays(datas, offsets)

        info = items[0].laser.info.copy()
        info["Name"] = "merge: " + info["Name"]
        info["Merge File Paths"] = ";".join(
            item.laser.info["File Path"] for item in items
        )

        laser = Laser(
            merge,
            calibration=items[0].laser.calibration,
            config=items[0].laser.config,
            info=info,
        )
        for item in items:
            self.graphics.scene().removeItem(item)
        self.addLaser(laser, pos=items[0].pos())

    def uniqueElements(self) -> list[str]:
        elements = set([])
        for item in self.laserItems():
            elements.update(item.laser.elements)
        return list(elements)

    def updateForItem(
        self,
        new: QtWidgets.QGraphicsItem,
        old: QtWidgets.QGraphicsItem | None = None,
        reason: QtCore.Qt.FocusReason = QtCore.Qt.NoFocusReason,
    ) -> None:
        # Todo: test if lock active
        if old == new:
            return
        if new is None:
            self.controls.setCurrentWidget(self.no_controls)
            return

        if isinstance(new, LaserImageItem):
            if isinstance(new, RGBLaserImageItem):
                self.controls.setCurrentWidget(self.rgb_laser_controls)
                self.rgb_laser_controls.setItem(new)
            else:
                self.controls.setCurrentWidget(self.laser_controls)
                self.laser_controls.setItem(new)
        elif isinstance(new, ImageOverlayItem):  # Todo: maybe add a proper class?
            self.controls.setCurrentWidget(self.image_controls)
            self.image_controls.setItem(new)
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
        self, pos: QtCore.QPointF, data_pos: QtCore.QPoint, v: float | np.ndarray | None
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
        else:
            v = np.atleast_1d(v)
            vstr = ",".join([f"{x:.4g}" if np.isfinite(x) else "nan" for x in v])
            status_bar.showMessage(f"{x:.4g},{y:.4g} [{vstr}]")

    # Callbacks
    def openDialog(
        self,
        dialog: str,
        item: ImageOverlayItem | LaserImageItem | None = None,
        selection: bool = False,
    ) -> QtWidgets.QDialog:
        if item is None:
            item = self.graphics.scene().focusItem()

        # Shared dialogs
        if dialog == "Pixel Size":
            dlg = dialogs.PixelSizeDialog(item.pixelSize(), parent=self)
            dlg.sizeSelected.connect(item.setPixelSize)
        elif dialog == "Transform":
            dlg = dialogs.TransformDialog(
                item.transform(), item.transformOriginPoint(), parent=self
            )
            dlg.transformChanged.connect(item.setTransform)
        elif not isinstance(item, LaserImageItem):
            raise ValueError(
                f"Item {item} is not a LaserImageItem, dialog {dialog} invalid."
            )
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
                (
                    {k: v.unit for k, v in item.laser.calibration.items()}
                    if item.options.calibrate
                    else {}
                ),
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
        elif tool == "Standards":
            widget = StandardsTool(item, view=self.view)
        else:
            raise ValueError(f"Invalid tool type {tool}.")

        widget.itemModified.connect(self.updateForItem)
        self.view.addTab(f"Tool: {tool}", widget)
        self.view.setActiveWidget(widget)
        return widget

    def dialogExport(self, item: LaserImageItem | None = None) -> QtWidgets.QDialog:
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
        self, item: LaserImageItem | None = None
    ) -> QtWidgets.QDialog | None:
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

        if len(self.laserItems()) > 1:
            menu.addAction(self.action_align_auto)
            menu.addAction(self.action_align_horz)
            menu.addAction(self.action_align_vert)
            menu.addSeparator()
            menu.addAction(self.action_merge)
        menu.popup(event.globalPos())
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.Close):
            for item in self.graphics.scene().selectedItems():
                item.close()
        elif event.matches(QtGui.QKeySequence.Cancel):
            self.graphics.endSelection()
            self.graphics.endWidget()
            self.graphics.endTransform()
        elif event.matches(QtGui.QKeySequence.Paste):
            mime = QtWidgets.QApplication.clipboard().mimeData()

            # Get all selected items, or the focussed item
            items = self.selectedLaserItems()
            if len(items) == 0 and isinstance(
                self.graphics.scene().focusItem(), LaserImageItem
            ):
                items.append(self.graphics.scene().focusItem())

            if mime.hasFormat("application/x-pew2laser"):
                with BytesIO(mime.data("application/x-pew2laser")) as fp:
                    data = np.load(fp)

                if mime.hasFormat("application/x-pew2config"):
                    with BytesIO(mime.data("application/x-pew2config")) as fp:
                        array = np.load(fp)
                        if array["spotsize"].size == 2:
                            config = SpotConfig.from_array(array)
                        else:
                            config = Config.from_array(array)
                else:
                    config = None

                if mime.hasFormat("application/x-pew2calibration"):
                    with BytesIO(mime.data("application/x-pew2calibration")) as fp:
                        npy = np.load(fp)
                        calibration = {k: Calibration.from_array(npy[k]) for k in npy}
                else:
                    calibration = None

                if mime.hasFormat("application/x-pew2info"):
                    with BytesIO(mime.data("application/x-pew2info")) as fp:
                        array = np.load(fp)
                        info = io.npz.unpack_info(array)
                else:
                    info = {"Name": "unknown"}

                laser = Laser(data, config=config, calibration=calibration, info=info)
                origin = self.graphics.mapFromGlobal(QtGui.QCursor.pos())
                self.addLaser(laser, self.graphics.mapToScene(origin))
            elif mime.hasFormat("application/x-pew2config"):
                with BytesIO(mime.data("application/x-pew2config")) as fp:
                    array = np.load(fp)
                    if array["spotsize"].size == 2:
                        config = SpotConfig.from_array(array)
                    else:
                        config = Config.from_array(array)
                for item in items:
                    item.applyConfig(config)
            elif mime.hasFormat("application/x-pew2calibration"):
                with BytesIO(mime.data("application/x-pew2calibration")) as fp:
                    npy = np.load(fp)
                    calibrations = {k: Calibration.from_array(npy[k]) for k in npy}
                for item in items:
                    item.applyCalibration(calibrations)

        super().keyPressEvent(event)
