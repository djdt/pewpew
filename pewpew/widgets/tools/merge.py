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
    closeRequested = QtCore.Signal("QListWidgetItem*")
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
        self.setDefaultDropAction(QtCore.Qt.TargetMoveAction)
        # self.indexesMoved.connect(lambda x: print(x, flush=True))
        self.setMovement(QtWidgets.QListView.Free)

        # self.model().rowsMoved.connect(lambda x: print("moved", x))
        # self.model().rowsAboutToBeInserted.connect(lambda x: print("inserted", x))
        # self.model().rowsAboutToBeRemoved.connect(lambda x, y ,z: print("removed", x, y ,z))

    # def supportedDropActions(self):
    #     return QtCore.Qt.CopyAction

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
        # self.list.indexesMoved.connect(lambda x: print("moved"))
        # self.list.rowsInserted.connect(lambda x: print("insert"))

        box_align = QtWidgets.QGroupBox("Align Images")
        layout_align = QtWidgets.QVBoxLayout()

        box_align.setLayout(layout_align)

        layout_graphics = QtWidgets.QVBoxLayout()
        layout_graphics.addWidget(self.graphics)

        layout_controls = QtWidgets.QVBoxLayout()
        layout_controls.addWidget(self.list, 1)
        layout_controls.addWidget(box_align, 0)

        self.box_graphics.setLayout(layout_graphics)
        self.box_controls.setLayout(layout_controls)


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
