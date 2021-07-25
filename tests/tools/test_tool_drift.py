import numpy as np

from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools.drift import DriftTool

from testing import linear_data


def test_tool_filter(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    widget = view.addLaser(Laser(linear_data(["a", "b", "c"])))
    tool = DriftTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    tool.combo_element.setCurrentText("a")
    tool.combo_element.activated.emit(0)

    tool.spinbox_degree.setValue(1)
    tool.apply()

    assert np.all(np.isclose(widget.laser.data["a"], widget.laser.data["a"][0][0]))

    tool.combo_element.setCurrentText("b")
    tool.combo_element.activated.emit(0)

    assert not np.all(np.isclose(widget.laser.data["b"], widget.laser.data["c"][0][0]))

    tool.check_apply_all.setChecked(True)
    tool.apply()

    assert np.all(np.isclose(widget.laser.data["b"], widget.laser.data["c"][0][0]))
