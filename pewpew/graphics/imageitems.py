from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np
import copy
from pathlib import Path

from pewlib import io
from pewlib.laser import Laser
from pewlib.calibration import Calibration
from pewlib.config import Config

# from pewlib.srr.config import SRRConfig

from pewpew.lib.numpyqt import array_to_image

from pewpew.graphics import colortable
from pewpew.graphics.options import GraphicsOptions

from pewpew.graphics.items import ColorBarItem, EditableLabelItem

from pewpew.actions import qAction

from typing import Any, Dict, List, Optional, Union


class SnapImageItem(QtWidgets.QGraphicsObject):
    selectionChanged = QtCore.Signal()
    imageChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QGraphicsItem] = None):
        super().__init__(parent)

        self.actions_order = [
            qAction(
                "object-order-front",
                "Send to Front",
                "Order the image in front of all others.",
                self.orderFirst,
            ),
            qAction(
                "object-order-raise",
                "Send Forwards",
                "Raise the images stacking order.",
                self.orderRaise,
            ),
            qAction(
                "object-order-lower",
                "Send Backwards",
                "Lower the images stacking order.",
                self.orderLower,
            ),
            qAction(
                "object-order-back",
                "Send to Back",
                "Order the image behind all others.",
                self.orderLast,
            ),
        ]

        self.action_close = qAction(
            "view-close", "Close", "Close the image.", self.close
        )

    def itemChange(
        self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value: Any
    ) -> Any:
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            pos = QtCore.QPointF(value)
            return self.snapPos(pos)
        return super().itemChange(change, value)

    def dataAt(self, pos: QtCore.QPointF) -> float:
        pos = self.mapToData(pos)
        return self.rawData()[pos.y(), pos.x()]

    def selectedAt(self, pos: QtCore.QPointF) -> bool:
        return False

    def imageSize(self) -> QtCore.QSize:
        raise NotImplementedError

    def rawData(self) -> np.ndarray:
        raise NotImplementedError

    def select(self, mask: np.ndarray, modes: List[str]) -> None:
        self.selectionChanged.emit()

    def mapToData(self, pos: QtCore.QPointF) -> QtCore.QPoint:
        """Map a position to an image pixel coordinate."""
        pixel = self.pixelSize()
        rect = self.boundingRect()
        return QtCore.QPoint(
            int((pos.x() - rect.left()) / pixel.width()),
            int((pos.y() - rect.top()) / pixel.height()),
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

    def close(self) -> None:
        self.deleteLater()
        if self.scene() is not None:
            self.scene().removeItem(self)

    def orderRaise(self) -> None:
        stack = [
            item
            for item in self.scene().items(
                self.sceneBoundingRect(), QtCore.Qt.IntersectsItemBoundingRect
            )
            if isinstance(item, SnapImageItem)
        ]
        idx = stack.index(self)
        if idx > 0:
            stack[idx - 1].stackBefore(self)

    def orderFirst(self) -> None:
        stack = [
            item
            for item in self.scene().items()
            if isinstance(item, SnapImageItem) and item is not self
        ]
        for item in stack:
            item.stackBefore(self)

    def orderLower(self) -> None:
        stack = [
            item
            for item in self.scene().items(
                self.sceneBoundingRect(), QtCore.Qt.IntersectsItemBoundingRect
            )
            if isinstance(item, SnapImageItem)
        ]
        idx = stack.index(self)
        if idx + 1 < len(stack):
            self.stackBefore(stack[idx + 1])

    def orderLast(self) -> None:
        stack = [
            item
            for item in self.scene().items()
            if isinstance(item, SnapImageItem) and item is not self
        ]
        for item in stack:
            self.stackBefore(item)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.LeftButton
            and self.flags() & QtWidgets.QGraphicsItem.ItemIsMovable
        ):
            self.setCursor(QtCore.Qt.SizeAllCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.setCursor(QtCore.Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


class ScaledImageItem(SnapImageItem):
    def __init__(
        self,
        image: QtGui.QImage,
        rect: QtCore.QRectF,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent=parent)
        self.setAcceptHoverEvents(True)
        self._last_hover_pos = QtCore.QPoint(-1, -1)

        self.image = image
        self.rect = rect

        self.action_copy_image = qAction(
            "insert-image",
            "Copy &Image",
            "Copy image to clipboard.",
            self.copyToClipboard,
        )

        self.action_pixel_size = qAction(
            "zoom-pixels",
            "Pixel Size",
            "Set the pixel width and height of the image.",
            None,
        )

    def imageSize(self) -> QtCore.QSize:
        return self.image.size()

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        painter.drawImage(self.rect, self.image)

    def copyToClipboard(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setImage(self.image)

    @classmethod
    def fromArray(
        cls,
        array: np.ndarray,
        rect: QtCore.QRectF,
        colortable: Optional[List[int]] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ) -> "ScaledImageItem":
        image = array_to_image(array)
        if colortable is not None:
            image.setColorTable(colortable)
            image.setColorCount(len(colortable))
        return cls(image, rect, parent)

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()

        menu.addActions(self.actions_order)
        menu.addSeparator()
        menu.addAction(self.action_close)

        menu.exec_(event.screenPos())
        event.accept()


class LaserImageItem(SnapImageItem):
    requestDialog = QtCore.Signal(str, QtWidgets.QGraphicsItem, bool)

    requestExport = QtCore.Signal(QtWidgets.QGraphicsItem)
    requestSave = QtCore.Signal(QtWidgets.QGraphicsItem)

    requestTool = QtCore.Signal(str, QtWidgets.QGraphicsItem)

    colortableChanged = QtCore.Signal(list, float, float, str)

    hoveredValueChanged = QtCore.Signal(QtCore.QPointF, QtCore.QPoint, float)
    hoveredValueCleared = QtCore.Signal()

    modified = QtCore.Signal()

    def __init__(
        self,
        laser: Laser,
        options: GraphicsOptions,
        current_element: Optional[str] = None,
        parent: Optional[QtWidgets.QGraphicsItem] = None,
    ):
        super().__init__(parent=parent)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsFocusable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._last_hover_pos = QtCore.QPoint(-1, -1)

        self.laser = laser
        self.options = options
        self.current_element = current_element or self.laser.elements[0]

        self.image: Optional[QtGui.QImage] = None
        self.mask_image: Optional[QtGui.QImage] = None

        self.raw_data: np.ndarray = np.array([])
        self.vmin, self.vmax = 0.0, 0.0

        # Name in top left corner
        self.element_label = EditableLabelItem(
            self,
            self.element(),
            "Element",
            font=self.options.font,
        )
        # self.element_label.labelChanged.connect(self.setElementName)
        self.label = EditableLabelItem(
            self,
            self.laser.info["Name"],
            "Laser Name",
            font=self.options.font,
            alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignRight,
        )
        self.label.labelChanged.connect(self.setName)

        self.colorbar = ColorBarItem(self, font=self.options.font)
        self.colorbar.setPos(self.boundingRect().bottomLeft())

        self.createActions()

    @property
    def mask(self) -> np.ndarray:
        if self.mask_image is None:
            return np.ones(self.laser.shape, dtype=bool)
        return self.mask_image._array.astype(bool)

    def element(self) -> str:
        return self.current_element

    def setElement(self, element: str) -> None:
        if element not in self.laser.elements:
            raise ValueError(
                f"Unknown element {element}. Expected one of {self.laser.elements}."
            )
        self.current_element = element
        self.element_label.setText(element)
        self.redraw()

    def name(self) -> str:
        return self.laser.info["Name"]

    def setName(self, name: str) -> None:
        self.laser.info["Name"] = name
        self.label.setText(name)
        self.modified.emit()

    # Virtual SnapImageItem methods
    def selectedAt(self, pos: QtCore.QPointF) -> bool:
        pos = self.mapToData(pos)
        return self.mask[pos.y(), pos.x()]

    def imageSize(self) -> QtCore.QSize:
        return QtCore.QSize(self.laser.shape[1], self.laser.shape[0])

    def rawData(self) -> np.ndarray:
        return self.raw_data

    def redraw(self) -> None:
        data = self.laser.get(
            self.element(), calibrate=self.options.calibrate, flat=True
        )
        unit = self.laser.calibration[self.element()].unit

        self.raw_data = np.ascontiguousarray(data)

        self.vmin, self.vmax = self.options.get_color_range_as_float(
            self.element(), self.raw_data
        )
        table = colortable.get_table(self.options.colortable)

        data = np.clip(self.raw_data, self.vmin, self.vmax)
        if self.vmin != self.vmax:  # Avoid div 0
            data = (data - self.vmin) / (self.vmax - self.vmin)

        self.image = array_to_image(data)
        self.image.setColorTable(table)
        self.image.setColorCount(len(table))

        # self.colortableChanged.emit(table, self.vmin, self.vmax, unit)
        self.colorbar.updateTable(table, self.vmin, self.vmax, unit)
        self.imageChanged.emit()
        self.update()

    def select(self, mask: np.ndarray, modes: List[str]) -> None:
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

        super().select(mask, modes)

    # GraphicsItem drawing
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

        if (
            self.hasFocus()
            and self.options.highlight_focus
            and not isinstance(painter.device(), QtGui.QPixmap)
        ):  # Only paint focus if option is active and not painting to a pixmap
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 127), 2.0, QtCore.Qt.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(self.boundingRect())

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

    def saveToFile(self, path: Union[Path, str]) -> None:
        path = Path(path)
        io.npz.save(path, self.laser)
        self.laser.info["File Path"] = str(path.resolve())

    # ==== Actions ===
    def createActions(self) -> None:
        self.action_copy_image = qAction(
            "insert-image",
            "Copy &Image",
            "Copy image to clipboard.",
            self.copyToClipboard,
        )

        # === IO requests ===
        self.action_export = qAction(
            "document-save-as",
            "E&xport",
            "Export laser data to a variety of formats.",
            lambda: self.requestExport.emit(self),
        )
        self.action_export.setShortcut("Ctrl+X")
        self.action_save = qAction(
            "document-save",
            "&Save",
            "Save document to numpy archive.",
            lambda: self.requestSave.emit(self),
        )
        self.action_save.setShortcut("Ctrl+S")

        # === Dialogs requests ===
        self.action_calibration = qAction(
            "go-top",
            "Ca&libration",
            "Edit the laser calibration.",
            lambda: self.requestDialog.emit("Calibration", self, False),
        )
        self.action_config = qAction(
            "document-edit",
            "&Config",
            "Edit the laser configuration.",
            lambda: self.requestDialog.emit("Config", self, False),
        )
        self.action_colocalisation = qAction(
            "dialog-information",
            "Colocalisation",
            "Open the colocalisation dialog.",
            lambda: self.requestDialog.emit("Colocalisation", self, False),
        )
        self.action_colocalisation_selection = qAction(
            "dialog-information",
            "Selection Colocalisation",
            "Open the colocalisation dialog for the current selected area.",
            lambda: self.requestDialog.emit("Colocalisation", self, True),
        )
        self.action_information = qAction(
            "documentinfo",
            "In&formation",
            "View and edit stored laser information.",
            lambda: self.requestDialog.emit("Information", self, False),
        )
        self.action_statistics = qAction(
            "dialog-information",
            "Statistics",
            "Open the statisitics dialog.",
            lambda: self.requestDialog.emit("Statistics", self, False),
        )
        self.action_statistics_selection = qAction(
            "dialog-information",
            "Selection Statistics",
            "Open the statisitics dialog for the current selected area.",
            lambda: self.requestDialog.emit("Statistics", self, True),
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
        self.action_show_colorbar = qAction(
            "visibility",
            "Show Colrbar",
            "Un-hide the colortable scale bar.",
            self.colorbar.show,
        )
        # self.action_duplicate = qAction(
        #     "edit-copy",
        #     "Duplicate image",
        #     "Open a copy of the image.",
        #     self.actionDuplicate,
        # )

        self.action_selection_copy_text = qAction(
            "insert-table",
            "Copy Selection as Text",
            "Copy the current selection to the clipboard as a column of text values.",
            self.copySelectionToText,
        )
        self.action_selection_crop = qAction(
            "transform-crop",
            "Crop to Selection",
            "Crop the image to the current selection.",
            self.cropToSelection,
        )

        self.actions_transform = [
            qAction(
                "object-flip-horizontal",
                "Flip Horizontal",
                "Flip data about the vertical axis.",
                lambda: self.transform(flip="horizontal"),
            ),
            qAction(
                "object-flip-vertical",
                "Flip Vertical",
                "Flip data about the horizontal axis.",
                lambda: self.transform(flip="vertical"),
            ),
            qAction(
                "object-rotate-left",
                "Rotate Left",
                "Rotate data 90 degrees counter-clockwise.",
                lambda: self.transform(rotate="left"),
            ),
            qAction(
                "object-rotate-right",
                "Rotate Right",
                "Rotate data 90 degrees clockwise.",
                lambda: self.transform(rotate="right"),
            ),
        ]

        self.actions_tools = [
            qAction(
                "folder-calculate",
                "Calculator",
                "Open the calculator tool for the current laser image.",
                lambda: self.requestTool.emit("Calculator", self),
            ),
            qAction(
                "view-filter",
                "Filter",
                "Apply various windowed filters to remove noise.",
                lambda: self.requestTool.emit("Filtering", self),
            ),
            qAction(
                "dialog-layers",
                "Overlay",
                "Tool for visualising multiple elements at once.",
                lambda: self.requestTool.emit("Overlay", self),
            ),
            qAction(
                "labplot-xy-fit-curve",
                "Standards",
                "Create calibrations from areas of the current laser.",
                lambda: self.requestTool.emit("Standards", self),
            ),
        ]

    def copySelectionToText(self) -> None:
        """Copies the currently selected data to the system clipboard."""
        data = self.raw_data[self.mask].ravel()

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

    def cropToSelection(self) -> None:
        """Crop image to current selection and open in a new tab.

        If selection is not rectangular then it is filled with nan.
        """
        raise NotImplementedError
        # mask = self.graphics.mask
        # if mask is None or np.all(mask == 0):  # pragma: no cover
        #     return
        # ix, iy = np.nonzero(mask)
        # x0, x1, y0, y1 = np.min(ix), np.max(ix) + 1, np.min(iy), np.max(iy) + 1

        # data = self.laser.data
        # new_data = np.empty((x1 - x0, y1 - y0), dtype=data.dtype)
        # for name in new_data.dtype.names:
        #     new_data[name] = np.where(
        #         mask[x0:x1, y0:y1], data[name][x0:x1, y0:y1], np.nan
        #     )

        # info = self.laser.info.copy()
        # info["Name"] = self.laserName() + "_cropped"
        # info["File Path"] = str(Path(info.get("File Path", "")).with_stem(info["Name"]))
        # new_widget = self.view.addLaser(
        #     Laser(
        #         new_data,
        #         calibration=self.laser.calibration,
        #         config=self.laser.config,
        #         info=info,
        #     )
        # )

        # new_widget.activate()

    def transform(
        self, flip: Optional[str] = None, rotate: Optional[str] = None
    ) -> None:
        """Transform the laser data.

        Args:
            flip: flip the image ['horizontal', 'vertical']
            rotate: rotate the image 90 degrees ['left', 'right']

        """
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

        self.prepareGeometryChange()
        self.redraw()

    # === Events ===
    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        mask_context = self.mask_image is not None and self.selectedAt(event.pos())

        menu = QtWidgets.QMenu()
        menu.addAction(self.action_copy_image)
        menu.addSeparator()

        menu.addAction(self.action_save)
        menu.addAction(self.action_export)

        menu.addSeparator()

        if mask_context:
            menu.addAction(self.action_selection_copy_text)
            menu.addAction(self.action_selection_crop)
            menu.addSeparator()

        if mask_context:
            menu.addAction(self.action_colocalisation_selection)
            menu.addAction(self.action_statistics_selection)
        else:
            menu.addAction(self.action_colocalisation)
            menu.addAction(self.action_statistics)

        menu.addSeparator()

        if not mask_context:
            transforms = menu.addMenu(
                QtGui.QIcon.fromTheme("transform-rotate"), "Transform"
            )
            transforms.addActions(self.actions_transform)
            menu.addSeparator()
            tools = menu.addMenu(QtGui.QIcon.fromTheme(""), "Tools")
            tools.addActions(self.actions_tools)
            menu.addSeparator()
            order = menu.addMenu(QtGui.QIcon.fromTheme(""), "Ordering")
            order.addActions(self.actions_order)

        menu.addAction(self.action_calibration)
        menu.addAction(self.action_config)
        menu.addAction(self.action_information)

        menu.addSection("words")
        menu.addSeparator()

        if not self.colorbar.isVisible():
            menu.addAction(self.action_show_colorbar)
        if not self.label.isVisible():
            menu.addAction(self.action_show_label_name)
        if not self.element_label.isVisible():
            menu.addAction(self.action_show_label_element)

        if not mask_context:
            menu.addSeparator()
            menu.addAction(self.action_close)

        menu.exec_(event.screenPos())
        event.accept()

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        pos = self.mapToData(event.pos())
        if pos != self._last_hover_pos:
            self._last_hover_pos = pos
            self.hoveredValueChanged.emit(
                event.pos(), pos, self.rawData()[pos.y(), pos.x()]
            )

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self._last_hover_pos = QtCore.QPoint(-1, -1)
        self.hoveredValueCleared.emit()
