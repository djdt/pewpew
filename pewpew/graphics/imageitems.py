import copy
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pewlib import io
from pewlib.calibration import Calibration
from pewlib.config import Config
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction
from pewpew.graphics import colortable
from pewpew.graphics.items import ColorBarItem, EditableLabelItem, RGBLabelItem
from pewpew.graphics.options import GraphicsOptions
from pewpew.lib.numpyqt import array_to_image, image_to_array


class SnapImageItem(QtWidgets.QGraphicsObject):
    selectionChanged = QtCore.Signal()
    imageChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QGraphicsItem | None = None):
        super().__init__(parent)

        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsFocusable
            | QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )

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
        self.action_close.setShortcut(QtGui.QKeySequence.Close)

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
        if self.scene() is not None:
            self.scene().removeItem(self)
        self.deleteLater()

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
        self.scene().update(self.boundingRect())

    def orderFirst(self) -> None:
        stack = [
            item
            for item in self.scene().items()
            if isinstance(item, SnapImageItem) and item is not self
        ]
        for item in stack:
            item.stackBefore(self)
        self.scene().update(self.boundingRect())

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
        self.scene().update(self.boundingRect())

    def orderLast(self) -> None:
        stack = [
            item
            for item in self.scene().items()
            if isinstance(item, SnapImageItem) and item is not self
        ]
        for item in stack:
            self.stackBefore(item)
        self.scene().update(self.boundingRect())

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Close):
            self.close()
        elif event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copyToClipboard()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.LeftButton
            and self.flags() & QtWidgets.QGraphicsItem.ItemIsMovable
        ):
            self.setCursor(QtCore.Qt.SizeAllCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.unsetCursor()
        super().mouseReleaseEvent(event)


class ScaledImageItem(SnapImageItem):
    def __init__(
        self,
        image: QtGui.QImage,
        rect: QtCore.QRectF,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(parent=parent)
        self.setAcceptHoverEvents(True)
        self._last_hover_pos = QtCore.QPoint(-1, -1)

        self.image = image
        self.rect = rect
        self._raw_data: np.ndarray | None = None

        self.action_copy_image = qAction(
            "insert-image",
            "Copy &Image",
            "Copy image to clipboard.",
            self.copyToClipboard,
        )

    def imageSize(self) -> QtCore.QSize:
        return self.image.size()

    def rawData(self) -> np.ndarray:
        if self._raw_data is None:
            self._raw_data = image_to_array(self.image)
        return self._raw_data

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.drawImage(self.rect, self.image)

        if self.isSelected() and not isinstance(
            painter.device(), QtGui.QPixmap
        ):  # Only paint focus if option is active and not painting to a pixmap
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 127), 2.0, QtCore.Qt.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(self.boundingRect())

    def copyToClipboard(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setImage(self.image)

    @classmethod
    def fromArray(
        cls,
        array: np.ndarray,
        rect: QtCore.QRectF,
        colortable: List[int] | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ) -> "ScaledImageItem":
        image = array_to_image(array)
        if colortable is not None:
            image.setColorTable(colortable)
            image.setColorCount(len(colortable))
        return cls(image, rect, parent)


class ImageOverlayItem(ScaledImageItem):
    requestDialog = QtCore.Signal(
        str, QtWidgets.QGraphicsItem, bool
    )  # type, self, use selection
    """Interactive ScaledImageItem class for overlay images."""

    def __init__(
        self,
        image: QtGui.QImage,
        rect: QtCore.QRectF,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(image, rect, parent=parent)

        self.action_pixel_size = qAction(
            "zoom-pixels",
            "Pixel Size",
            "Set the pixel width and height of the image.",
            lambda: self.requestDialog.emit("Pixel Size", self, False),
        )
        self.action_lock = qAction(
            "folder-locked",
            "Lock Image",
            "Locks the image, preventing interaction.",
            self.lock,
        )
        # self.action_transform_scale = qAction(
        #     "transform-scale",
        #     "Scale Image",
        #     "Scale the image.",
        #     self.startScale,
        # )
        self.action_unlock = qAction(
            "folder-unlocked",
            "Unlock Image",
            "unlocks the image, allowing interaction.",
            self.unlock,
        )

    def lock(self) -> None:
        """Locking is performed by preventing focus and use of left mouse button."""
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(
            self.acceptedMouseButtons() & (~QtCore.Qt.LeftButton)
        )

    def unlock(self) -> None:
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(self.acceptedMouseButtons() | QtCore.Qt.LeftButton)

    def isLocked(self) -> bool:
        return not self.acceptedMouseButtons() & QtCore.Qt.LeftButton

    def setPixelSize(self, size: QtCore.QSizeF) -> None:
        image_size = self.imageSize()
        self.prepareGeometryChange()
        self.rect = QtCore.QRectF(
            self.rect.topLeft(),
            QtCore.QSizeF(
                size.width() * image_size.width(), size.height() * image_size.height()
            ),
        )

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()

        menu.addAction(self.action_copy_image)
        menu.addAction(self.action_pixel_size)
        menu.addSeparator()
        menu.addActions(self.actions_order)
        menu.addSeparator()
        if self.isLocked():
            menu.addAction(self.action_unlock)
        else:
            menu.addAction(self.action_lock)
        menu.addSeparator()
        menu.addAction(self.action_close)

        menu.exec(event.screenPos())
        event.accept()


class LaserImageItem(SnapImageItem):
    requestDialog = QtCore.Signal(
        str, QtWidgets.QGraphicsItem, bool
    )  # type, self, use selection

    requestAddLaser = QtCore.Signal(Laser)
    requestExport = QtCore.Signal(QtWidgets.QGraphicsItem)
    requestSave = QtCore.Signal(QtWidgets.QGraphicsItem)

    requestTool = QtCore.Signal(str, QtWidgets.QGraphicsItem)
    requestConversion = QtCore.Signal(str, QtWidgets.QGraphicsItem)

    colortableChanged = QtCore.Signal(list, float, float, str)
    elementsChanged = QtCore.Signal()

    hoveredValueChanged = QtCore.Signal(QtCore.QPointF, QtCore.QPoint, np.ndarray)
    hoveredValueCleared = QtCore.Signal()

    modified = QtCore.Signal()

    def __init__(
        self,
        laser: Laser,
        options: GraphicsOptions,
        current_element: str | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        super().__init__(parent=parent)
        self.setAcceptHoverEvents(True)

        self._last_hover_pos = QtCore.QPoint(-1, -1)

        self.laser = laser
        self.options = options
        self.current_element = current_element or self.laser.elements[0]

        self.image: QtGui.QImage | None = None
        self.mask_image: QtGui.QImage | None = None

        self.raw_data: np.ndarray = np.array([])
        self.vmin, self.vmax = 0.0, 0.0
        # Name in top left corner
        self.element_label = EditableLabelItem(
            self,
            self.element(),
            "Element",
            font=self.options.font,
        )
        self.element_label.labelChanged.connect(self.setElementName)
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

    def setElementName(self, new: str) -> None:
        names = {n: n for n in self.laser.elements}
        names[self.current_element] = new
        self.renameElements(names)

    def renameElements(self, names: Dict[str, str]) -> None:
        old_names = [x for x in self.laser.elements if x not in names]
        self.laser.remove(old_names)
        self.laser.rename(names)
        if self.current_element in names:
            self.setElement(names[self.current_element])
        else:
            self.setElement(self.laser.elements[0])
        self.elementsChanged.emit()
        self.modified.emit()

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
        self.raw_data = np.ascontiguousarray(data)
        self.vmin, self.vmax = self.options.get_color_range_as_float(
            self.element(), self.raw_data
        )
        data = np.clip(self.raw_data, self.vmin, self.vmax)
        if self.vmin != self.vmax:  # Avoid div 0
            data = (data - self.vmin) / (self.vmax - self.vmin)

        unit = self.laser.calibration[self.element()].unit
        table = colortable.get_table(self.options.colortable)
        table[0] = self.options.nan_color.rgba()

        self.image = array_to_image(data)
        self.image.setColorTable(table)
        self.image.setColorCount(len(table))

        # self.colortableChanged.emit(table, self.vmin, self.vmax, unit)
        self.colorbar.updateTable(table, self.vmin, self.vmax, unit)
        # Update the colorbar position in case the image aspect has changed
        self.colorbar.setPos(self.boundingRect().bottomLeft())
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
        widget: QtWidgets.QWidget | None = None,
    ):
        painter.save()
        if self.options.smoothing:
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        rect = self.boundingRect()

        if self.image is not None:
            painter.drawImage(rect, self.image)

        if self.mask_image is not None:
            painter.drawImage(rect, self.mask_image)

        if self.isSelected() and not isinstance(
            painter.device(), QtGui.QPixmap
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
        mime = QtCore.QMimeData()
        with BytesIO() as fp:
            np.save(fp, self.laser.data)
            mime.setData("application/x-pew2laser", fp.getvalue())
        with BytesIO() as fp:
            np.save(fp, self.laser.config.to_array())
            mime.setData("application/x-pew2config", fp.getvalue())
        with BytesIO() as fp:
            np.savez(fp, **{k: v.to_array() for k, v in self.laser.calibration.items()})
            mime.setData("application/x-pew2calibration", fp.getvalue())
        with BytesIO() as fp:
            np.save(fp, io.npz.pack_info(self.laser.info, remove_keys=[]))
            mime.setData("application/x-pew2info", fp.getvalue())
        clipboard.setMimeData(mime)

    def copyImageToClipboard(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setImage(self.image)

    def saveToFile(self, path: Path | str) -> None:
        path = Path(path)
        io.npz.save(path, self.laser)
        self.laser.info["File Path"] = str(path.resolve())

    # ==== Actions ===
    def createActions(self) -> None:
        self.action_copy = qAction(
            "edit-copy",
            "Copy",
            "Copy item to clipboard.",
            self.copyToClipboard,
        )
        self.action_copy.setShortcut(QtGui.QKeySequence.Copy)
        self.action_copy_image = qAction(
            "insert-image",
            "Copy &Image",
            "Copy current image to clipboard.",
            self.copyImageToClipboard,
        )

        # === IO requests ===
        self.action_export = qAction(
            "document-save-as",
            "E&xport",
            "Export laser data to a variety of formats.",
            lambda: self.requestExport.emit(self),
        )
        self.action_export.setShortcut(QtGui.QKeySequence.StandardKey.Cut)
        self.action_save = qAction(
            "document-save",
            "&Save",
            "Save document to numpy archive.",
            lambda: self.requestSave.emit(self),
        )
        self.action_save.setShortcut(QtGui.QKeySequence.Save)

        self.action_convert_rgb = qAction(
            "adjustrgb",
            "Convert to RGB",
            "Change the image to an RGB type display.",
            lambda: self.requestConversion.emit("RGBLaserImageItem", self),
        )

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
        if self.mask_image is None:
            return
        mask = self.mask
        ix, iy = np.nonzero(mask)
        x0, x1, y0, y1 = np.min(ix), np.max(ix) + 1, np.min(iy), np.max(iy) + 1

        data = self.laser.data
        new_data = np.empty((x1 - x0, y1 - y0), dtype=data.dtype)
        for name in new_data.dtype.names:
            new_data[name] = np.where(
                mask[x0:x1, y0:y1], data[name][x0:x1, y0:y1], np.nan
            )

        info = self.laser.info.copy()
        info["Name"] = info["Name"] + "_cropped"
        info["File Path"] = str(Path(info.get("File Path", "")).with_stem(info["Name"]))
        self.requestAddLaser.emit(
            Laser(
                new_data,
                calibration=self.laser.calibration,
                config=self.laser.config,
                info=info,
            )
        )

    def transform(self, flip: str | None = None, rotate: str | None = None) -> None:
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
        menu.addAction(self.action_copy)
        menu.addAction(self.action_copy_image)
        menu.addSeparator()

        menu.addAction(self.action_save)
        menu.addAction(self.action_export)

        menu.addSeparator()

        menu.addAction(self.action_convert_rgb)

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

        menu.addSeparator()

        if not self.colorbar.isVisible() and self.colorbar.isEnabled():
            menu.addAction(self.action_show_colorbar)
        if not self.label.isVisible() and self.label.isEnabled():
            menu.addAction(self.action_show_label_name)
        if not self.element_label.isVisible() and self.element_label.isEnabled():
            menu.addAction(self.action_show_label_element)

        if not mask_context:
            menu.addSeparator()
            menu.addAction(self.action_close)

        menu.exec_(event.screenPos())
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Save):
            self.requestSave.emit(self)
        elif event.matches(QtGui.QKeySequence.StandardKey.Cut):
            self.requestExport.emit(self)
        else:
            super().keyPressEvent(event)

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


class RGBLaserImageItem(LaserImageItem):
    requestExport = QtCore.Signal(QtWidgets.QGraphicsItem)
    requestSave = QtCore.Signal(QtWidgets.QGraphicsItem)

    requestTool = QtCore.Signal(str, QtWidgets.QGraphicsItem)

    colortableChanged = QtCore.Signal(list, float, float, str)

    hoveredValueChanged = QtCore.Signal(QtCore.QPointF, QtCore.QPoint, np.ndarray)
    hoveredValueCleared = QtCore.Signal()

    modified = QtCore.Signal()

    class RGBElement(object):
        def __init__(
            self,
            element: str,
            color: QtGui.QColor,
            prange: Tuple[float, float] = (0.0, 99.0),
        ):
            self.element = element
            self.color = color
            self.prange = prange

        def __repr__(self) -> str:
            return f"RGBElement({self.element}, {self.color!r}, {self.prange})"

    def __init__(
        self,
        laser: Laser,
        options: GraphicsOptions,
        current_elements: List[RGBElement] | None = None,
        parent: QtWidgets.QGraphicsItem | None = None,
    ):
        if current_elements is None:
            colors = [
                QtGui.QColor(255, 0, 0),
                QtGui.QColor(0, 255, 0),
                QtGui.QColor(0, 0, 255),
            ]
            current_elements = [
                RGBLaserImageItem.RGBElement(element, color, (0.0, 99.0))
                for element, color in zip(laser.elements[:3], colors)
            ]
        super().__init__(laser, options, current_elements[0].element)
        # self.setAcceptHoverEvents(False)

        # Redo action
        self.action_convert_rgb.setText("Convert to Colortable")
        self.action_convert_rgb.triggered.disconnect()
        self.action_convert_rgb.triggered.connect(
            lambda: self.requestConversion.emit("LaserImageItem", self)
        )

        # Disable colorbar
        self.colorbar.setVisible(False)
        self.colorbar.setEnabled(False)
        self.element_label.setVisible(False)
        self.element_label.setEnabled(False)

        self.subtractive = False
        self.current_elements: List[RGBLaserImageItem.RGBElement] = current_elements

        self.elements_label = RGBLabelItem(
            self,
            [rgb.element for rgb in self.current_elements],
            [rgb.color for rgb in self.current_elements],
            font=self.options.font,
            alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft,
        )

    def redraw(self) -> None:
        if len(self.current_elements) == 0:
            self.raw_data == np.zeros((*self.laser.shape[:2], 3))
        else:
            self.raw_data = np.stack(
                [
                    self.laser.get(
                        element.element, calibrate=self.options.calibrate, flat=True
                    )
                    for element in self.current_elements[:3]
                ],
                axis=2,
            )
        data = np.zeros((*self.laser.shape[:2], 3))
        for i, element in enumerate(self.current_elements):
            if element.element not in self.laser.elements:
                continue
            rgb = np.array(element.color.getRgbF()[:3])
            if self.subtractive:
                rgb = 1.0 - rgb

            # Normalise to range
            vmin, vmax = np.percentile(self.raw_data[:, :, i], element.prange)
            x = np.clip(self.raw_data[:, :, i], vmin, vmax)
            if vmin != vmax:
                x = (x - vmin) / (vmax - vmin)
            # Convert to separate rgb channels
            data += x[:, :, None] * rgb

        if self.subtractive:
            data = 1.0 - data

        self.image = array_to_image(data)

        self.imageChanged.emit()
        self.update()

    def setElement(self, element: str) -> None:
        if element not in self.laser.elements:
            raise ValueError(
                f"Unknown element {element}. Expected one of {self.laser.elements}."
            )

        self.current_elements[0].element = element
        super().setElement(element)

    def renameElements(self, names: Dict[str, str]) -> None:
        super().renameElements(names)
        for rgb, element in zip(self.current_elements, self.laser.elements[:3]):
            rgb.element = element

    def setCurrentElements(self, elements: List[RGBElement]) -> None:
        self.current_elements = elements
        self.elements_label.setTexts([rgb.element for rgb in elements])
        self.elements_label.colors = [rgb.color for rgb in elements]
        if len(self.current_elements) > 0:
            self.setElement(self.current_elements[0].element)
        else:
            self.redraw()

    @classmethod
    def fromLaserImageItem(
        cls,
        item: LaserImageItem,
        options: GraphicsOptions,
    ) -> "RGBLaserImageItem":
        pmin, pmax = options.get_color_range_as_percentile(
            item.current_element, item.raw_data
        )
        return cls(
            Laser(
                item.laser.data.copy(),
                config=item.laser.config,
                calibration=item.laser.calibration,
                info=item.laser.info,
            ),
            options,
            parent=item.parent,
        )
