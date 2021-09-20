from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from pewlib.laser import Laser
from pewlib.process.register import fft_register_images

from pewpew.actions import qAction, qToolButton

from pewpew.lib.numpyqt import array_to_image

from pewpew.graphics import colortable
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.imageitems import ScaledImageItem

from pewpew.graphics.overlaygraphics import OverlayScene, OverlayView

from pewpew.widgets.tools import ToolWidget
from pewpew.widgets.laser import LaserWidget


from typing import Dict, List, Optional, Tuple

# class MergeImage(ScaledImageItem, QtWidgets.QGraphicsObject):
#     pass


class MergeGraphicsView(OverlayView):
    def __init__(self, options: GraphicsOptions, parent: QtWidgets.QWidget = None):
        self.options = options
        self._scene = OverlayScene(0, 0, 640, 480)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        super().__init__(scene=self._scene, parent=parent)

        self.setInteractionFlag("tool")

    def drawImage(
        self, data: np.ndarray, rect: QtCore.QRect, name: str
    ) -> ScaledImageItem:
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
        scaled_image = ScaledImageItem(image, rect, smooth=False, snap=True)
        scaled_image.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.scene().addItem(scaled_image)

        return scaled_image

    def fitAllImages(self) -> None:
        images = filter(lambda x: isinstance(x, ScaledImageItem), self.scene().items())

        union = QtCore.QRectF(0, 0, 0, 0)
        for image in images:
            union = union.united(image.rect.translated(image.pos()))
        self.scene().setSceneRect(union)
        self.fitInView(union, QtCore.Qt.KeepAspectRatio)


class MergeRowItem(QtWidgets.QWidget):
    elementChanged = QtCore.Signal("QWidget*", str)
    closeRequested = QtCore.Signal("QWidget*")

    def __init__(
        self,
        laser: Laser,
        item: QtWidgets.QListWidgetItem,
        parent: "MergeLaserList",
    ):
        super().__init__(parent)

        self.laser = laser
        self.image: Optional[ScaledImageItem] = None
        self.item = item

        self.action_close = qAction(
            "window-close", "Remove", "Remove laser.", self.close
        )

        self.label_name = QtWidgets.QLabel(laser.info["Name"])
        self.label_offset = QtWidgets.QLabel("(0.0, 0.0)")
        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.addItems(laser.elements)

        self.combo_element.currentTextChanged.connect(self.comboElementChanged)

        self.button_close = qToolButton(action=self.action_close)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label_name, 0)
        layout.addWidget(self.label_offset, 0)
        layout.addStretch(1)
        layout.addWidget(self.combo_element, 0)
        layout.addWidget(self.button_close, 0)

        self.setLayout(layout)

    def data(self, calibrate: bool = False) -> np.ndarray:
        return self.laser.get(
            self.combo_element.currentText(), flat=True, layer=None, calibrate=calibrate
        )

    def extentRect(self) -> QtCore.QRectF:
        x0, x1, y0, y1 = self.laser.extent
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

    def updateOffset(self) -> None:
        if self.image is None:
            return
        pos = self.image.pos()
        self.label_offset.setText(f"({pos.x():.1f}, {pos.y():.1f})")

    def comboElementChanged(self, name: str) -> None:
        self.elementChanged.emit(self, name)

    def close(self) -> None:
        if self.image is not None:
            self.image.scene().removeItem(self.image)
        self.closeRequested.emit(self.item)
        super().close()


class MergeLaserList(QtWidgets.QListWidget):
    elementChanged = QtCore.Signal("QWidget*")

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

    def addRow(self, laser: Laser) -> QtWidgets.QListWidgetItem:
        item = QtWidgets.QListWidgetItem(self)
        self.addItem(item)

        row = MergeRowItem(laser, item, parent=self)
        row.elementChanged.connect(self.elementChanged)
        row.closeRequested.connect(self.removeRow)

        item.setSizeHint(row.sizeHint())
        self.setItemWidget(item, row)
        return item

    def removeRow(self, item: QtWidgets.QListWidgetItem) -> None:
        self.takeItem(self.row(item))


class MergeTool(ToolWidget):
    """Tool for merging laser images."""

    normalise_methods = ["Maximum", "Minimum"]

    def __init__(self, widget: LaserWidget):
        super().__init__(widget, apply_all=False)

        self.graphics = MergeGraphicsView(self.viewspace.options, parent=self)

        self.list = MergeLaserList()
        self.list.elementChanged.connect(self.redrawRow)

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

    def redrawRow(self, row: MergeRowItem) -> None:
        pos = QtCore.QPointF(0.0, 0.0)
        if row.image is not None:
            pos = row.image.pos()
            self.graphics.scene().removeItem(row.image)
            row.image = None

        data = row.data(calibrate=self.graphics.options.calibrate)
        row.image = self.graphics.drawImage(data, row.extentRect(), row.combo_element.currentText())
        row.image.setPos(pos)

        row.image.xChanged.connect(row.updateOffset)
        row.image.yChanged.connect(row.updateOffset)

        self.graphics.scene().addItem(row.image)

    def refresh(self) -> None:
        for row in self.list.rows:
            self.redrawRow(row)

        self.graphics.fitAllImages()
        super().refresh()

    def alignImagesFFT(self) -> None:
        rows = self.list.rows
        if len(rows) == 0:
            return
        base = rows[0].data()
        rows[0].image.setPos(QtCore.QPointF(0.0, 0.0))

        for row in rows[1:]:
            offset = fft_register_images(base, row.data())
            w, h = (
                row.laser.config.get_pixel_width(),
                row.laser.config.get_pixel_height(),
            )
            row.image.setPos(QtCore.QPointF(offset[0] * w, offset[1] * h))

        self.refresh()

    def alignImagesLeftToRight(self) -> None:
        sum = 0.0
        for row in self.list.rows:
            row.image.setPos(QtCore.QPointF(sum, 0.0))
            sum += row.image.rect.width()

        self.refresh()

    def alignImagesTopToBottom(self) -> None:
        sum = 0.0
        for row in self.list.rows:
            row.image.setPos(QtCore.QPointF(0.0, sum))
            sum += row.image.rect.height()

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
