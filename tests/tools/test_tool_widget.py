from pewlib.laser import Laser
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot
from testing import rand_data

from pewpew.widgets.laser import LaserTabView
from pewpew.widgets.tools.tool import ToolWidget


def test_tool_widget(qtbot: QtBot):
    view = LaserTabView()
    qtbot.addWidget(view)
    view.show()

    widget = view.importFile(Laser(rand_data(["a", "b"]), info={"Name": "test"}))
    item = widget.laserItems()[0]
    tool = ToolWidget(item, apply_all=True)
    view.addTab("Tool", tool)
    with qtbot.waitExposed(tool):
        tool.show()

    tool.requestClose()

    with qtbot.wait_signal(tool.applyPressed):
        button = tool.button_box.button(QtWidgets.QDialogButtonBox.Apply)
        button.click()

    with qtbot.wait_signal(tool.applyPressed):
        button = tool.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        button.click()
