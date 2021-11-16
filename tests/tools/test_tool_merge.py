import numpy as np

from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools.merge import MergeTool

from testing import rand_data


def test_merge_tool(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()

    data = rand_data(["A", "B"])
    view.addLaser(Laser(data, info={"Name": "Laser 1", "File Path": "/test/laser1"}))
    data = rand_data(["B", "C"])
    view.addLaser(Laser(data, info={"Name": "Laser 2", "File Path": "/test/laser1"}))

    tool = MergeTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitExposed(tool)

    assert tool.list.count() == 1

    # Test load via dialog
    dlg = tool.addLaserDialog()
    assert dlg.comboBoxItems() == ["Laser 2"]
    dlg.accept()

    assert tool.list.count() == 2
    assert [row.offset() == (0, 0) for row in tool.list.rows]
    assert tool.list.rows[0].combo_element.currentText() == "A"
    assert tool.list.rows[1].combo_element.currentText() == "B"

    tool.action_align_horz.trigger()

    assert tool.list.rows[0].offset() == (0, 0)
    assert tool.list.rows[1].offset() == (0, 10)

    tool.action_align_vert.trigger()

    assert tool.list.rows[0].offset() == (0, 0)
    assert tool.list.rows[1].offset() == (10, 0)

    tool.action_align_auto.trigger()

    assert tool.list.rows[0].offset() == (0, 0)
    # Second image position unknown

    tool.apply()
