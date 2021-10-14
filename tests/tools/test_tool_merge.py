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
    view.addLaser(Laser(data, info={"Name": "Laser 1"}))
    data = rand_data(["B", "C"])
    view.addLaser(Laser(data, info={"Name": "Laser 2"}))

    tool = MergeTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitExposed(tool)
