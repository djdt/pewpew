import numpy as np

from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools.filtering import FilteringTool

from testing import rand_data


def test_tool_filter(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    view.addLaser(Laser(rand_data(["a"])))
    tool = FilteringTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitExposed(tool)

    tool.combo_filter.setCurrentText("Mean")
    tool.combo_filter.activated.emit(0)
    tool.lineedit_fparams[0].setText("3.0")
    tool.lineedit_fparams[1].setText("3.0")
    tool.lineedit_fparams[0].editingFinished.emit()
    assert np.all(tool.fparams == [3.0, 3.0])
    assert tool.isComplete()

    tool.lineedit_fparams[0].setText("5.0")
    tool.lineedit_fparams[0].editingFinished.emit()
    assert np.all(tool.fparams == [5.0, 3.0])
    assert tool.isComplete()

    tool.combo_filter.setCurrentText("Median")
    tool.combo_filter.activated.emit(0)
    assert np.all(tool.fparams == [5.0, 3.0])
    assert tool.isComplete()

    tool.lineedit_fparams[0].setText("4.0")
    tool.lineedit_fparams[0].editingFinished.emit()
    assert not tool.isComplete()

    tool.combo_filter.setCurrentText("Mean")
    tool.combo_filter.activated.emit(0)
    assert np.all(tool.fparams == [5.0, 3.0])
    assert tool.isComplete()

    tool.apply()
