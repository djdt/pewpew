import numpy as np

from PySide2 import QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools.tool import ToolWidget

from testing import rand_data


def test_tool_widget(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()

    view = viewspace.activeView()
    widget = view.addLaser(Laser(rand_data("A1"), name="Widget"))
    tool = ToolWidget(widget, apply_all=True)
    index = widget.index

    widget.view.removeTab(index)
    widget.view.insertTab(index, "Tool", tool)
    qtbot.waitForWindowShown(tool)

    tool.requestClose()
    view.tabs.tabText(index) == "Widget"

    with qtbot.wait_signal(tool.applyPressed):
        button = tool.button_box.button(QtWidgets.QDialogButtonBox.Apply)
        button.click()

    with qtbot.wait_signal(tool.applyPressed):
        button = tool.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        button.click()
