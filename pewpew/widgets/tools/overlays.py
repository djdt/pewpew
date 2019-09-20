from PySide2 import QtWidgets

from pewpew.widgets.canvases import BasicCanvas
from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.tools import Tool


class OverlayCanvas(BasicCanvas):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)


class OverlayTool(Tool):
    def __init__(self, widget: LaserWidget, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.widget = widget
        self.canvas = OverlayCanvas()
