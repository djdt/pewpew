from PySide2 import QtCore, QtGui, QtWidgets

from pewlib.laser import Laser

from pewpew.actions import qAction, qToolButton

from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.imageitems import ScaledImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView

from pewpew.widgets.tools import ToolWidget
from pewpew.widgets.laser import LaserWidget

from typing import List, Optional, Tuple


class MergeGraphicsView(LaserGraphicsView):
    def __init__(self, options: GraphicsOptions, parent: QtWidgets.QWidget = None):
        super().__init__(options, parent=parent)
        self.setInteractionFlag("tool")

        self.merge_images: List[ScaledImageItem] = []


class MergeRowItem(QtWidgets.QWidget):
    closeRequested = QtCore.Signal("QWidget*")
    itemChanged = QtCore.Signal()

    def __init__(
        self, laser: Laser, item: QtWidgets.QListWidgetItem, parent: "MergeLaserList"
    ):
        super().__init__(parent)

        self.laser = laser
        self.item = item

        self.action_close = qAction(
            "window-close", "Remove", "Remove laser.", self.close
        )

        self.label_name = QtWidgets.QLabel(self.laser.info["Name"])
        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.addItems(self.laser.elements)

        self.combo_element.currentIndexChanged.connect(self.itemChanged)

        self.button_close = qToolButton(action=self.action_close)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label_name, 0)
        layout.addStretch(1)
        layout.addWidget(self.combo_element, 0)
        layout.addWidget(self.button_close, 0)

        self.setLayout(layout)

    def close(self) -> None:
        self.closeRequested.emit(self.item)
        super().close()


class MergeLaserList(QtWidgets.QListWidget):
    itemChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        # Allow reorder
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

    @property
    def lasers(self) -> List[Laser]:
        return [self.itemWidget(self.item(i)).laser for i in range(self.count())]

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
        row.itemChanged.connect(self.itemChanged)
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
        self.list.itemChanged.connect(self.refresh)

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
        lasers = self.list.lasers

        super().refresh()

    def alignImagesFFT(self) -> None:
        pass
    def alignImagesLeftToRight(self) -> None:
        pass
    def alignImagesTopToBottom(self) -> None:
        pass

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
    app.exec_()
