from typing import List
from PySide2 import QtWidgets

from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.imageitems import ScaledImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView

from pewpew.widgets.tools import ToolWidget
from pewpew.widgets.laser import LaserWidget


class MergeGraphicsView(LaserGraphicsView):
    def __init__(self, options: GraphicsOptions, parent: QtWidgets.QWidget = None):
        super().__init__(options, parent=parent)
        self.setInteractionFlag("tool")

        self.merge_images: List[ScaledImageItem] = []


class MergeTool(ToolWidget):
    """Tool for merging laser images."""

    normalise_methods = ["Maximum", "Minimum"]

    def __init__(self, widget: LaserWidget):
        super().__init__(widget, apply_all=False)

        self.graphics = MergeGraphicsView(self.viewspace.options, parent=self)

        self.combo_base_element = QtWidgets.QComboBox()

        layout_graphics = QtWidgets.QVBoxLayout()
        layout_graphics.addWidget(self.graphics)

        layout_controls = QtWidgets.QVBoxLayout()

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
    view.activeView().removeTab(0)
    view.activeView().insertTab(0, "", tool)
    view.show()
    app.exec_()
