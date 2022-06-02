from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from pewlib.laser import Laser
from pewlib.process.register import fft_register_offset, overlap_structured_arrays

from pewpew.actions import qAction, qToolButton

from pewpew.lib.numpyqt import array_to_image

from pewpew.graphics import colortable
from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.imageitems import SnapImageItem

from pewpew.graphics.overlaygraphics import OverlayGraphicsView

from pewpew.widgets.tools import ToolWidget
from pewpew.widgets.laser import LaserTabWidget


from typing import List, Optional, Tuple

# TODO: possible off by one on overlap
# shared range
# Alpha
# Add / mult


class MergeGraphicsView(OverlayGraphicsView):
    def __init__(self, options: GraphicsOptions, parent: Optional[QtWidgets.QWidget] = None):
        self.options = options
        self._scene = QtWidgets.QGraphicsScene(-1e6, -1e6, 2e6, 2e6)
        self._scene.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))
        super().__init__(scene=self._scene, parent=parent)

        self.setInteractionFlag("tool")

    def drawImage(
        self, data: np.ndarray, rect: QtCore.QRect, name: str
    ) -> SnapImageItem:
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
        scaled_image = SnapImageItem(image, rect, smooth=False, snap=True)
        scaled_image.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.scene().addItem(scaled_image)

        return scaled_image

    def fitAllImages(self) -> None:
        images = filter(lambda x: isinstance(x, SnapImageItem), self.scene().items())

        union = QtCore.QRectF(0, 0, 0, 0)
        for image in images:
            union = union.united(image.rect.translated(image.pos()))
        # self.scene().setSceneRect(union)
        self.fitInView(union, QtCore.Qt.KeepAspectRatio)


class MergeRowItem(QtWidgets.QWidget):
    alphaChanged = QtCore.Signal("QWidget*")
    elementChanged = QtCore.Signal("QWidget*")
    closeRequested = QtCore.Signal("QWidget*")

    def __init__(
        self,
        laser: Laser,
        item: QtWidgets.QListWidgetItem,
        close_button: bool,
        parent: "MergeLaserList",
    ):
        super().__init__(parent)

        self.laser = laser
        self.image: Optional[SnapImageItem] = None
        self.item = item

        self.action_close = qAction(
            "window-close", "Remove", "Remove laser.", self.close
        )

        self.label_name = QtWidgets.QLabel(laser.info["Name"])
        self.label_offset = QtWidgets.QLabel("(0.0, 0.0)")
        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.addItems(laser.elements)

        self.combo_element.currentTextChanged.connect(
            lambda _: self.elementChanged.emit(self.item)
        )

        self.button_close = qToolButton(action=self.action_close)
        self.button_close.setEnabled(close_button)

        self.slider_alpha = QtWidgets.QSlider()
        self.slider_alpha.setMinimumWidth(200)
        self.slider_alpha.setOrientation(QtCore.Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(100)
        self.slider_alpha.valueChanged.connect(
            lambda _: self.alphaChanged.emit(self.item)
        )

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label_name, 2)
        layout.addWidget(self.label_offset, 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.slider_alpha, 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.combo_element, 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.button_close, 0, QtCore.Qt.AlignRight)

        self.setLayout(layout)

    def alpha(self) -> float:
        return self.slider_alpha.value() / 100.0

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

    def offset(self) -> Tuple[int, int]:
        if self.image is None:
            return (0, 0)
        pos = self.image.pos()
        size = self.image.pixelSize()
        return (int(pos.y() / size.height()), int(pos.x() / size.width()))

    def comboElementChanged(self, name: str) -> None:
        self.elementChanged.emit(self.item, name)

    def close(self) -> None:
        if self.image is not None:
            self.image.scene().removeItem(self.image)
        self.closeRequested.emit(self.item)
        super().close()


class MergeLaserList(QtWidgets.QListWidget):
    rowAlphaChanged = QtCore.Signal("QWidget*")
    rowElementChanged = QtCore.Signal("QWidget*")

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
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

    def addRow(
        self, laser: Laser, close_button: bool = True
    ) -> QtWidgets.QListWidgetItem:
        item = QtWidgets.QListWidgetItem(self)
        self.addItem(item)

        row = MergeRowItem(laser, item, close_button=close_button, parent=self)
        row.elementChanged.connect(self.rowElementChanged)
        row.alphaChanged.connect(self.rowAlphaChanged)
        row.closeRequested.connect(self.removeRow)

        item.setSizeHint(row.sizeHint())
        self.setItemWidget(item, row)
        return item

    def removeRow(self, item: QtWidgets.QListWidgetItem) -> None:
        self.takeItem(self.row(item))


class MergeTool(ToolWidget):
    """Tool for merging laser images."""

    normalise_methods = ["Maximum", "Minimum"]

    def __init__(self, widget: LaserTabWidget):
        super().__init__(widget, orientation=QtCore.Qt.Vertical, apply_all=False)
        self.button_box.removeButton(
            self.button_box.button(QtWidgets.QDialogButtonBox.Apply)
        )

        self.graphics = MergeGraphicsView(self.viewspace.options, parent=self)
        self.graphics.setMouseTracking(True)

        self.list = MergeLaserList()
        self.list.rowElementChanged.connect(self.redrawRow)
        self.list.rowAlphaChanged.connect(self.updateRowAlpha)
        self.list.model().rowsMoved.connect(self.reassignZValues)

        self.button_add = QtWidgets.QPushButton("Add Laser")
        self.button_add.clicked.connect(self.addLaserDialog)

        box_align = QtWidgets.QGroupBox("Align Images")
        box_align.setLayout(QtWidgets.QVBoxLayout())

        self.action_align_auto = qAction(
            "view-refresh",
            "FFT Register",
            "Register all images to the topmost image.",
            self.alignImagesFFT,
        )
        self.action_align_horz = qAction(
            "align-vertical-top",
            "Left to Right",
            "Layout images in a horizontal line.",
            self.alignImagesLeftToRight,
        )
        self.action_align_vert = qAction(
            "align-horizontal-left",
            "Top to Bottom",
            "Layout images in a vertical line.",
            self.alignImagesTopToBottom,
        )

        self.button_align = qToolButton("align-horizontal-left", "Align Images")
        self.button_align.addAction(self.action_align_auto)
        self.button_align.addAction(self.action_align_horz)
        self.button_align.addAction(self.action_align_vert)

        layout_right = QtWidgets.QVBoxLayout()
        layout_right.addWidget(self.button_align)
        layout_right.addStretch(1)
        layout_right.addWidget(self.button_add)

        layout_graphics = QtWidgets.QVBoxLayout()
        layout_graphics.addWidget(self.graphics)

        layout_controls = QtWidgets.QHBoxLayout()
        layout_controls.addWidget(self.list, 1)
        # layout_controls.addWidget(self.button_add, 0)
        layout_controls.addLayout(layout_right)

        self.box_graphics.setLayout(layout_graphics)
        self.box_controls.setLayout(layout_controls)

        # Add the tool widgets laser, make unclosable
        self.list.addRow(self.widget.laser, close_button=False)

        self.refresh()

    def apply(self) -> None:
        count = self.list.count()
        if count < 2:
            return
        base = self.list.rows[0]
        # Align to first row
        merge = base.laser.get(calibrate=False)
        for row in self.list.rows[1:]:
            offset = np.array(row.offset()) - base.offset()
            merge = overlap_structured_arrays(
                merge, row.laser.get(calibrate=False), offset=offset
            )

        info = base.laser.info.copy()
        info["Name"] = "merge: " + info["Name"]
        info["Merge File Paths"] = ";".join(
            row.laser.info["File Path"] for row in self.list.rows
        )

        # Merge calibrations

        laser = Laser(
            merge,
            calibration=base.laser.calibration,
            config=base.laser.config,
            info=info,
        )
        self.view.addLaser(laser)

    def addLaserDialog(self) -> QtWidgets.QInputDialog:
        lasers = [
            w.laser
            for view in self.viewspace.views
            for w in view.widgets()
            if isinstance(w, LaserTabWidget)
        ]
        current_lasers = [row.laser for row in self.list.rows]
        lasers = [laser for laser in lasers if laser not in current_lasers]

        if len(lasers) == 0:
            return

        dlg = QtWidgets.QInputDialog(self)
        dlg.setComboBoxItems([laser.info["Name"] for laser in lasers])
        dlg.setLabelText("Laser:")
        dlg.setWindowTitle("Select Laser")
        dlg.textValueSelected.connect(self.addLaserByName)
        dlg.open()

        return dlg

    def addLaserByName(self, name: str) -> None:
        lasers = [
            w.laser
            for view in self.viewspace.views
            for w in view.widgets()
            if isinstance(w, LaserTabWidget)
        ]
        current_lasers = [row.laser for row in self.list.rows]
        lasers = [laser for laser in lasers if laser not in current_lasers]
        if len(lasers) == 0:
            return
        laser = [laser for laser in lasers if laser.info["Name"] == name][0]

        item = self.list.addRow(laser)

        self.redrawRow(item)
        self.graphics.fitAllImages()

    def updateRowAlpha(self,item: QtWidgets.QListWidgetItem) -> None: 
        row = self.list.itemWidget(item)
        if row.image is not None:
            alpha = row.alpha()
            row.image.setOpacity(alpha)

    def redrawRow(self, item: QtWidgets.QListWidgetItem) -> None:
        pos = QtCore.QPointF(0.0, 0.0)
        row = self.list.itemWidget(item)
        if row.image is not None:
            pos = row.image.pos()
            self.graphics.scene().removeItem(row.image)
            row.image = None

        data = row.data(calibrate=self.graphics.options.calibrate)
        row.image = self.graphics.drawImage(
            data, row.extentRect(), row.combo_element.currentText()
        )
        alpha = row.alpha()

        row.image.setOpacity(alpha)
        row.image.setPos(pos)
        row.image.setZValue(self.list.row(item))

        row.image.xChanged.connect(row.updateOffset)
        row.image.yChanged.connect(row.updateOffset)

    def reassignZValues(
        self,
        source: QtCore.QModelIndex,
        start: int,
        end: int,
        dest: QtCore.QModelIndex,
        row: int,
    ) -> None:
        for i in range(self.list.count()):
            w = self.list.itemWidget(self.list.item(i))
            w.image.setZValue(-i)

    def refresh(self) -> None:
        for i in range(self.list.count()):
            self.redrawRow(self.list.item(i))

        self.graphics.fitAllImages()
        super().refresh()

    def alignImagesFFT(self) -> None:
        rows = self.list.rows
        if len(rows) == 0:
            return
        base = rows[0].data()
        rows[0].image.setPos(QtCore.QPointF(0.0, 0.0))

        for row in rows[1:]:
            offset = fft_register_offset(base, row.data())
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