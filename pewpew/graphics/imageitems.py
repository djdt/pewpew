from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np
import copy

from pewlib.laser import Laser
from pewlib.calibration import Calibration
from pewlib.config import Config

# from pewlib.srr.config import SRRConfig

from pewpew.lib.numpyqt import array_to_image

from pewpew.graphics import colortable
from pewpew.graphics.options import GraphicsOptions

# from pewpew.graphics.overlayitems import OverlayItem, LabelOverlay
from pewpew.graphics.items import EditableLabelItem

from pewpew.actions import qAction

from typing import Any, Dict, List, Optional


class SnapImageItem(QtWidgets.QGraphicsObject):
    selectionChanged = QtCore.Signal()
    imageChanged = QtCore.Signal()

    def itemChange(
        self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value: Any
    ) -> Any:
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            pos = QtCore.QPointF(value)
            size = self.pixelSize()
            pos.setX(pos.x() - pos.x() % size.width())
            pos.setY(pos.y() - pos.y() % size.height())
            return pos
        return super().itemChange(change, value)

    def dataAtPos(self, pos: QtCore.QPointF) -> float:
        raise NotImplementedError

    def imageSize(self) -> QtCore.QSize:
        raise NotImplementedError

    def rawData(self) -> np.ndarray:
        raise NotImplementedError

    def handleSelection(self, mask: np.ndarray, modes: List[str]) -> None:
        self.selectionChanged.emit()

    def mapToData(self, pos: QtCore.QPointF) -> QtCore.QPoint:
        """Map a position to an image pixel coordinate."""
        pixel = self.pixelSize()
        rect = self.boundingRect()
        return QtCore.QPoint(
            (pos.x() - rect.left()) / pixel.width(),
            (pos.y() - rect.top()) / pixel.height(),
        )

    def pixelSize(self) -> QtCore.QSizeF:
        """Size / scaling of an image pixel."""
        rect = self.boundingRect()
        size = self.imageSize()
        return QtCore.QSizeF(rect.width() / size.width(), rect.height() / size.height())

    def snapPos(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        pixel = self.pixelSize()
        x = round(pos.x() / pixel.width()) * pixel.width()
        y = round(pos.y() / pixel.height()) * pixel.height()
        return QtCore.QPointF(x, y)


class LaserImageItem(SnapImageItem):
    requestDialogCalibration = QtCore.Signal()
    requestDialogConfig = QtCore.Signal()
    requestDialogColocalisation = QtCore.Signal()
    requestDialogInformation = QtCore.Signal()
    requestDialogStatistics = QtCore.Signal()
    colortableChanged = QtCore.Signal(list, float, float)

    modified = QtCore.Signal()

    def __init__(
        self,
        laser: Laser,
        options: GraphicsOptions,
        current_element: Optional[str] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent=parent)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)

        self.laser = laser
        self.options = options
        self.element = current_element or self.laser.elements[0]

        self.image: Optional[QtGui.QImage] = None
        self.mask_image: Optional[QtGui.QImage] = None

        self.raw_data: np.ndarray = np.array([])
        self.vmin, self.vmax = 0.0, 0.0

        # Name in top left corner
        self.element_label = EditableLabelItem(
            self,
            self.element,
            font=self.options.font,
        )
        self.label = EditableLabelItem(
            self,
            self.laser.info["Name"],
            font=self.options.font,
            alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignRight,
        )
        self.label.labelChanged.connect(self.rename)

        self.createActions()

    def setElement(self, element: str) -> None:
        if element not in self.laser.elements:
            raise ValueError(
                f"Unknown element {element}. Expected one of {self.laser.elements}."
            )
        self.element = element
        self.element_label.setText(element)
        self.redraw()

    @property
    def name(self) -> str:
        return self.laser.info["Name"]

    @name.setter
    def name(self, name: str) -> None:
        self.laser.info["Name"] = name
        self.modified.emit()

    @property
    def mask(self) -> np.ndarray:
        if self.mask_image is None:
            return np.ones(self.laser.shape, dtype=bool)
        return self.mask_image._array.astype(bool)

    def rename(self, name: str) -> None:
        self.laser.info["Name"] = name
        self.label.setText(name)

    # Virtual SnapImageItem methods
    def dataAtPos(self, pos: QtCore.QPointF) -> float:
        pos = self.mapToData(pos)
        return self.raw_data[pos.y(), pos.x()]

    def imageSize(self) -> QtCore.QSize:
        return QtCore.QSize(self.laser.shape[1], self.laser.shape[0])

    def rawData(self) -> np.ndarray:
        return self.raw_data

    def redraw(self) -> None:
        data = self.laser.get(self.element, calibrate=self.options.calibrate, flat=True)
        self.raw_data = np.ascontiguousarray(data)

        self.vmin, self.vmax = self.options.get_color_range_as_float(
            self.element, self.raw_data
        )
        table = colortable.get_table(self.options.colortable)

        data = np.clip(self.raw_data, self.vmin, self.vmax)
        if self.vmin != self.vmax:  # Avoid div 0
            data = (data - self.vmin) / (self.vmax - self.vmin)

        self.image = array_to_image(data)
        self.image.setColorTable(table)
        self.image.setColorCount(len(table))

        self.colortableChanged.emit(table, self.vmin, self.vmax)
        self.imageChanged.emit()

    def handleSelection(self, mask: np.ndarray, modes: List[str]) -> None:
        current_mask = self.mask

        if "add" in modes:
            mask = np.logical_or(current_mask, mask)
        elif "subtract" in modes:
            mask = np.logical_and(current_mask, ~mask)
        elif "intersect" in modes:
            mask = np.logical_and(current_mask, mask)
        elif "difference" in modes:
            mask = np.logical_xor(current_mask, mask)

        color = QtGui.QColor(255, 255, 255, a=128)

        if np.any(mask):
            self.mask_image = array_to_image(mask.astype(np.uint8))
            self.mask_image.setColorTable([0, int(color.rgba())])
            self.mask_image.setColorCount(2)
        else:
            self.mask_image = None

        self.update()

        super().handleSelection(mask, modes)

    def boundingRect(self) -> QtCore.QRectF:
        x0, x1, y0, y1 = self.laser.config.data_extent(self.laser.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        rect.moveTopLeft(QtCore.QPointF(0, 0))
        return rect

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ):
        painter.save()
        if self.options.smoothing:
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        rect = self.boundingRect()

        if self.image is not None:
            painter.drawImage(rect, self.image)

        if self.mask_image is not None:
            painter.drawImage(rect, self.mask_image)

        # if self.isSelected():
        #     painter.setPen(QtGui.QPen(QtCore.Qt.white, 10.0))
        #     painter.drawRect(self.boundingRect())

        # if self.hasFocus():
        #     painter.setRenderHint(QtGui.QPainter.Antialiasing)
        #     fm = QtGui.QFontMetrics(self.options.font, painter.device())
        #     path = QtGui.QPainterPath()
        #     path.addText(0, fm.ascent(), self.options.font, self.laser.info["Name"])

        #     # painter.setTransform
        #     # painter.setBrushOrigin(rect.topRight())
        #     # painter.setTransform(QtGui.QTransform.fromTranslate(rect.left(), rect.top()))

        #     painter.setRenderHint(QtGui.QPainter.Antialiasing)
        #     painter.strokePath(path, QtGui.QPen(QtCore.Qt.black, 2.0))
        #     painter.fillPath(
        #         path, QtGui.QBrush(self.options.font_color, QtCore.Qt.SolidPattern)
        #     )
        # #     self.rect = self.laserRect()
        # #     self.image = self.laserImage(self.current_element)
        painter.restore()

    # === Slots ===
    def applyCalibration(self, calibrations: Dict[str, Calibration]) -> None:
        """Set laser calibrations."""
        modified = False
        for element in calibrations:
            if element in self.laser.calibration:
                self.laser.calibration[element] = copy.copy(calibrations[element])
                modified = True
        if modified:
            self.modified.emit()
            self.redraw()

    def applyConfig(self, config: Config) -> None:
        """Set laser configuration."""
        # Only apply if the type of config is correct
        if type(config) is type(self.laser.config):  # noqa
            self.laser.config = copy.copy(config)
            self.modified.emit()
            self.redraw()

    def applyInformation(self, info: Dict[str, str]) -> None:
        """Set laser information."""
        if self.laser.info != info:
            self.laser.info = info
            self.modified.emit()

    def copyToClipboard(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setImage(self.image)

    # ==== Actions ===
    def createActions(self) -> None:
        self.action_copy_image = qAction(
            "insert-image",
            "Copy &Image",
            "Copy image to clipboard.",
            self.copyToClipboard,
        )
        # === Dialogs requests ===
        self.action_calibration = qAction(
            "go-top",
            "Ca&libration",
            "Edit the laser calibration.",
            self.requestDialogCalibration,
        )
        self.action_config = qAction(
            "document-edit",
            "&Config",
            "Edit the laser configuration.",
            self.requestDialogConfig,
        )
        self.action_colocalisation = qAction(
            "dialog-information",
            "Colocalisation",
            "Open the colocalisation dialog.",
            self.requestDialogColocalisation,
        )
        self.action_information = qAction(
            "documentinfo",
            "In&formation",
            "View and edit stored laser information.",
            self.requestDialogInformation,
        )
        self.action_statistics = qAction(
            "dialog-information",
            "Statistics",
            "Open the statisitics dialog.",
            self.requestDialogStatistics,
        )

        self.action_show_label_name = qAction(
            "visibility",
            "Show Name Label",
            "Un-hide the laser name label.",
            self.label.show,
        )
        self.action_show_label_element = qAction(
            "visibility",
            "Show Element Label",
            "Un-hide the current element label.",
            self.element_label.show,
        )
        # self.action_duplicate = qAction(
        #     "edit-copy",
        #     "Duplicate image",
        #     "Open a copy of the image.",
        #     self.actionDuplicate,
        # )
        # self.action_export = qAction(
        #     "document-save-as", "E&xport", "Export documents.", self.actionExport
        # )
        # self.action_export.setShortcut("Ctrl+X")
        # # Add the export action so we can use it via shortcut
        # self.addAction(self.action_export)
        # self.action_save = qAction(
        #     "document-save", "&Save", "Save document to numpy archive.", self.actionSave
        # )
        # self.action_save.setShortcut("Ctrl+S")
        # Add the save action so we can use it via shortcut
        # self.addAction(self.action_save)
        # self.action_select_statistics = qAction(
        #     "dialog-information",
        #     "Selection Statistics",
        #     "Open the statisitics dialog for the current selection.",
        #     self.actionStatisticsSelection,
        # )
        # self.action_select_colocalisation = qAction(
        #     "dialog-information",
        #     "Selection Colocalisation",
        #     "Open the colocalisation dialog for the current selection.",
        #     self.actionColocalSelection,
        # )

        # self.action_select_copy_text = qAction(
        #     "insert-table",
        #     "Copy Selection as Text",
        #     "Copy the current selection to the clipboard as a column of text values.",
        #     self.actionCopySelectionText,
        # )
        # self.action_select_crop = qAction(
        #     "transform-crop",
        #     "Crop to Selection",
        #     "Crop the image to the current selection.",
        #     self.actionCropSelection,
        # )

    # def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
    #     if self.label.boundingRect().contains(event.scenePos()):
    #         self.label.mouseDoubleClickEvent(event)
    #     super().mouseDoubleClickEvent(event)

    # def actionCalibration(self) -> QtWidgets.QDialog:
    #     """Open a `:class:pewpew.widgets.dialogs.CalibrationDialog` and applies result."""
    #     dlg = dialogs.CalibrationDialog(
    #         self.laser.calibration, self.current_element, parent=self
    #     )
    #     dlg.calibrationSelected.connect(self.applyCalibration)
    #     dlg.calibrationApplyAll.connect(self.view.applyCalibration)
    #     dlg.open()
    #     return dlg

    # def actionConfig(self) -> QtWidgets.QDialog:
    #     """Open a `:class:pewpew.widgets.dialogs.ConfigDialog` and applies result."""
    #     dlg = dialogs.ConfigDialog(self.laser.config, parent=self)
    #     dlg.configSelected.connect(self.applyConfig)
    #     dlg.configApplyAll.connect(self.view.applyConfig)
    #     dlg.open()
    #     return dlg

    # def actionCopyImage(self) -> None:
    #     self.graphics.copyToClipboard()

    # def actionCopySelectionText(self) -> None:
    #     """Copies the currently selected data to the system clipboard."""
    #     data = self.graphics.data[self.graphics.mask].ravel()

    #     html = (
    #         '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
    #         "<table>"
    #     )
    #     text = ""
    #     for x in data:
    #         html += f"<tr><td>{x:.10g}</td></tr>"
    #         text += f"{x:.10g}\n"
    #     html += "</table>"

    #     mime = QtCore.QMimeData()
    #     mime.setHtml(html)
    #     mime.setText(text)
    #     QtWidgets.QApplication.clipboard().setMimeData(mime)

    # def actionCropSelection(self) -> None:
    #     self.cropToSelection()

    # def actionDuplicate(self) -> None:
    #     """Duplicate document to a new tab."""
    #     self.view.addLaser(copy.deepcopy(self.laser))

    # def actionExport(self) -> QtWidgets.QDialog:
    #     """Opens a `:class:pewpew.exportdialogs.ExportDialog`.

    #     This can save the document to various formats.
    #     """
    #     dlg = exportdialogs.ExportDialog(self, parent=self)
    #     dlg.open()
    #     return dlg

    # def actionInformation(self) -> QtWidgets.QDialog:
    #     """Opens a `:class:pewpew.widgets.dialogs.InformationDialog`."""
    #     dlg = dialogs.InformationDialog(self.laser.info, parent=self)
    #     dlg.infoChanged.connect(self.applyInformation)
    #     dlg.open()
    #     return dlg

    # def actionRequestColorbarEdit(self) -> None:
    #     if self.viewspace is not None:
    #         self.viewspace.colortableRangeDialog()

    # def actionSave(self) -> QtWidgets.QDialog:
    #     """Save the document to an '.npz' file.

    #     If not already associated with an '.npz' path a dialog is opened to select one.
    #     """
    #     path = Path(self.laser.info["File Path"])
    #     if path.suffix.lower() == ".npz" and path.exists():
    #         self.saveDocument(path)
    #         return None
    #     else:
    #         path = self.laserFilePath()
    #     dlg = QtWidgets.QFileDialog(
    #         self, "Save File", str(path.resolve()), "Numpy archive(*.npz);;All files(*)"
    #     )
    #     dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    #     dlg.fileSelected.connect(self.saveDocument)
    #     dlg.open()
    #     return dlg

    # def actionSelectDialog(self) -> QtWidgets.QDialog:
    #     """Open a `:class:pewpew.widgets.dialogs.SelectionDialog` and applies selection."""
    #     dlg = dialogs.SelectionDialog(self.graphics, parent=self)
    #     dlg.maskSelected.connect(self.graphics.drawSelectionImage)
    #     self.refreshed.connect(dlg.refresh)
    #     dlg.show()
    #     return dlg

    # def actionStatistics(self, crop_to_selection: bool = False) -> QtWidgets.QDialog:
    #     """Open a `:class:pewpew.widgets.dialogs.StatsDialog` with image data.

    #     Args:
    #         crop_to_selection: pass current selection as a mask
    #     """
    #     data = self.laser.get(calibrate=self.graphics.options.calibrate, flat=True)
    #     mask = self.graphics.mask
    #     if mask is None or not crop_to_selection:
    #         mask = np.ones(data.shape, dtype=bool)

    #     units = {}
    #     if self.graphics.options.calibrate:
    #         units = {k: v.unit for k, v in self.laser.calibration.items()}

    #     dlg = dialogs.StatsDialog(
    #         data,
    #         mask,
    #         units,
    #         self.current_element,
    #         pixel_size=(
    #             self.laser.config.get_pixel_width(),
    #             self.laser.config.get_pixel_height(),
    #         ),
    #         parent=self,
    #     )
    #     dlg.open()
    #     return dlg

    # def actionStatisticsSelection(self) -> QtWidgets.QDialog:
    #     return self.actionStatistics(True)

    # def actionColocal(self, crop_to_selection: bool = False) -> QtWidgets.QDialog:
    #     """Open a `:class:pewpew.widgets.dialogs.ColocalisationDialog` with image data.

    #     Args:
    #         crop_to_selection: pass current selection as a mask
    #     """
    #     data = self.laser.get(flat=True)
    #     mask = self.graphics.mask if crop_to_selection else None

    #     dlg = dialogs.ColocalisationDialog(data, mask, parent=self)
    #     dlg.open()
    #     return dlg

    # def actionColocalSelection(self) -> QtWidgets.QDialog:
    #     return self.actionColocal(True)

    # === Events ===
    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneEvent) -> None:
        self.setSelected(True)

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()
        # menu.addAction(self.action_duplicate)
        menu.addAction(self.action_copy_image)
        menu.addSeparator()
        # menu.addAction(self.action_calibration)

        # if self.graphics.posInSelection(event.pos()):
        #     menu.addAction(self.action_select_copy_text)
        #     menu.addAction(self.action_select_crop)
        #     menu.addSeparator()
        #     menu.addAction(self.action_select_statistics)
        #     menu.addAction(self.action_select_colocalisation)
        # else:
        #     menu.addAction(self.view.action_open)
        #     menu.addAction(self.action_save)
        #     menu.addAction(self.action_export)
        #     menu.addSeparator()
        menu.addAction(self.action_config)
        menu.addAction(self.action_calibration)
        menu.addAction(self.action_information)

        menu.addSeparator()

        if not self.label.isVisible():
            menu.addAction(self.action_show_label_name)
        if not self.element_label.isVisible():
            menu.addAction(self.action_show_label_element)

        #     menu.addSeparator()
        #     menu.addAction(self.action_statistics)
        #     menu.addAction(self.action_colocalisation)
        menu.exec_(event.screenPos())
        event.accept()
