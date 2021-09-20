from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from pewlib.laser import Laser
from pewlib.process.register import fft_register_images

from pewpew.actions import qAction, qToolButton

from pewpew.lib.numpyqt import array_to_image

from pewpew.graphics import colortable
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.imageitems import ScaledImageItem

# from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.overlaygraphics import OverlayScene, OverlayView

from pewpew.widgets.tools import ToolWidget
from pewpew.widgets.laser import LaserWidget


from typing import List, Optional, Tuple


class MergeGraphicsView(OverlayView):
    def __init__(self, options: GraphicsOptions, parent: QtWidgets.QWidget = None):
        self.options = options
        self._scene = OverlayScene(0, 0, 640, 480)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        super().__init__(scene=self._scene, parent=parent)

        self.setInteractionFlag("tool")

        self.merge_images: List[ScaledImageItem] = []

    def drawImage(self, data: np.ndarray, rect: QtCore.QRect, name: str) -> None:
        """Draw 'data' into 'rect'.

        Args:
            data: image data
            rect: image extent
            name: label of data
        """
        self.data = np.ascontiguousarray(data)

        vmin, vmax = self.options.get_colorrange_as_float(name, self.data)
        table = colortable.get_table(self.options.colortable)

        data = np.clip(self.data, vmin, vmax)
        if vmin != vmax:  # Avoid div 0
            data = (data - vmin) / (vmax - vmin)

        image = array_to_image(data)
        image.setColorTable(table)
        scaled_image = ScaledImageItem(image, rect, smooth=self.options.smoothing)
        self.scene().addItem(scaled_image)
        self.merge_images.append(scaled_image)

        # self.colorbar.updateTable(table, vmin, vmax)

    def boundingRect(self) -> QtCore.QRectF:
        if len(self.merge_images) == 0:
            return QtCore.QRectF(0.0, 0.0, 1.0, 1.0)

        union = self.merge_images[0].rect
        for image in self.merge_images[1:]:
            union = union.united(image.rect)
        return union

    # def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
    #     super().mouseMoveEvent(event)
    #     pos = self.mapToScene(event.pos())
    #     if (
    #         self.image is not None
    #         and self.image.rect.left() < pos.x() < self.image.rect.right()
    #         and self.image.rect.top() < pos.y() < self.image.rect.bottom()
    #     ):
    #         dpos = self.mapToData(pos)
    #         self.cursorValueChanged.emit(
    #             pos.x(), pos.y(), self.data[dpos.y(), dpos.x()]
    #         )
    #     else:
    #         self.cursorValueChanged.emit(pos.x(), pos.y(), np.nan)


class MergeRowItem(QtWidgets.QWidget):
    closeRequested = QtCore.Signal("QWidget*")
    elementChanged = QtCore.Signal(str)

    def __init__(
        self, laser: Laser, item: QtWidgets.QListWidgetItem, parent: "MergeLaserList"
    ):
        super().__init__(parent)

        self.laser = laser
        self.item = item
        self.offset = (0.0, 0.0)

        self.action_close = qAction(
            "window-close", "Remove", "Remove laser.", self.close
        )

        self.label_name = QtWidgets.QLabel(self.laser.info["Name"])
        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.addItems(self.laser.elements)

        self.combo_element.currentIndexChanged.connect(self.elementChanged)

        self.button_close = qToolButton(action=self.action_close)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label_name, 0)
        layout.addStretch(1)
        layout.addWidget(self.combo_element, 0)
        layout.addWidget(self.button_close, 0)

        self.setLayout(layout)

    def data(self, calibrate: bool = False) -> np.ndarray:
        return self.laser.get(
            self.combo_element.currentText(), calibrate=calibrate, layer=None, flat=True
        )

    def rect(self) -> QtCore.QRectF:
        x0, x1, y0, y1 = self.laser.extent
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        return rect.translated(self.offset[0], self.offset[1])

    def close(self) -> None:
        self.closeRequested.emit(self.item)
        super().close()


class MergeLaserList(QtWidgets.QListWidget):
    refreshRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        # Allow reorder
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

    @property
    def rows(self) -> List[MergeRowItem]:
        return [self.itemWidget(self.item(i)) for i in range(self.count())]

    def dropMimeData(
        self,
        index: int,
        data: QtCore.QMimeData,
        action: QtCore.Qt.DropAction,
    ) -> bool:
        if action != QtCore.Qt.TargetMoveAction:
            return False
        return super().dropMimeData(index, data, action)

    def addRow(self, laser: Laser) -> None:
        item = QtWidgets.QListWidgetItem(self)
        self.addItem(item)

        row = MergeRowItem(laser, item, self)
        row.elementChanged.connect(self.refreshRequested)
        row.closeRequested.connect(self.removeRow)

        item.setSizeHint(row.sizeHint())
        self.setItemWidget(item, row)

    def removeRow(self, item: QtWidgets.QListWidgetItem) -> None:
        self.takeItem(self.row(item))


class MergeTool(ToolWidget):
    """Tool for merging laser images."""

    normalise_methods = ["Maximum", "Minimum"]

    def __init__(self, widget: LaserWidget):
        super().__init__(widget, apply_all=False)

        self.graphics = MergeGraphicsView(self.viewspace.options, parent=self)

        self.list = MergeLaserList()
        self.list.refreshRequested.connect(self.refresh)
        self.list.model().rowsRemoved.connect(self.refresh)

        box_align = QtWidgets.QGroupBox("Align Images")
        box_align.setLayout(QtWidgets.QVBoxLayout())

        action_align_auto = qAction(
            "view-refresh",
            "FFT Register",
            "Register all images to the topmost image.",
            self.alignImagesFFT,
        )
        action_align_horz = qAction(
            "align-vertical-top",
            "Left to Right",
            "Layout images in a horizontal line.",
            self.alignImagesLeftToRight,
        )
        action_align_vert = qAction(
            "align-horizontal-left",
            "Top to Bottom",
            "Layout images in a vertical line.",
            self.alignImagesTopToBottom,
        )

        for action in [action_align_auto, action_align_horz, action_align_vert]:
            button = qToolButton(action=action)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

            box_align.layout().addWidget(button)

        layout_graphics = QtWidgets.QVBoxLayout()
        layout_graphics.addWidget(self.graphics)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addWidget(self.list, 1)
        layout_controls.addWidget(box_align, 0)

        self.box_graphics.setLayout(layout_graphics)
        self.box_controls.setLayout(layout_controls)

    def refresh(self) -> None:
        # Clear current images
        for image in self.graphics.merge_images:
            self.graphics.scene().removeItem(image)
        self.graphics.merge_images.clear()

        for row in self.list.rows:
            data = row.data(calibrate=self.graphics.options.calibrate)
            rect = row.rect()

            self.graphics.drawImage(data, rect, row.combo_element.currentText())

        # Set the view to contain all images
        rect = self.graphics.boundingRect()
        self.graphics.setSceneRect(rect)
        self.graphics.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        super().refresh()

    def alignImagesFFT(self) -> None:
        rows = self.list.rows
        if len(rows) == 0:
            return
        base = rows[0].data()
        rows[0].offset = (0.0, 0.0)

        for row in rows[1:]:
            offset = fft_register_images(base, row.data())
            w, h = row.laser.config.get_pixel_width(), row.laser.config.get_pixel_height()
            row.offset = (offset[0] * w, offset[1] * h)

        self.refresh()

    def alignImagesLeftToRight(self) -> None:
        sum = 0.0
        for row in self.list.rows:
            row.offset = (0.0, sum)
            sum += row.rect().height()

        self.refresh()

    def alignImagesTopToBottom(self) -> None:
        sum = 0.0
        for row in self.list.rows:
            row.offset = (sum, 0.0)
            sum += row.rect().width()

        self.refresh()


if __name__ == "__main__":
    import pewlib.io
    from pewpew.widgets.laser import LaserViewSpace

    app = QtWidgets.QApplication()
    view = LaserViewSpace()
    laser1 = pewlib.io.npz.load(
        "/home/tom/MEGA/Uni/Experimental/LAICPMS/2019 Micro Arrays/20190617_krisskross/20190617_me401c_r4c3.npz"
    )
    widget = view.activeView().addLaser(laser1)
    tool = MergeTool(widget)
    tool.list.addRow(widget.laser)
    tool.list.addRow(widget.laser)
    tool.list.addRow(widget.laser)
    view.activeView().removeTab(0)
    view.activeView().insertTab(0, "", tool)
    view.show()
    view.resize(800, 600)
    tool.refresh()
    app.exec_()
