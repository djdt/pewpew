import numpy as np

from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.widgets.laser import LaserTabView
from pewpew.widgets.tools.filtering import FilteringTool

from testing import rand_data


def test_tool_filter(qtbot: QtBot):
    view = LaserTabView()
    qtbot.addWidget(view)
    view.show()

    widget = view.importFile(Laser(rand_data(["a", "b"]), info={"Name": "test"}))
    item = widget.laserItems()[0]
    tool = FilteringTool(item)
    view.addTab("Tool", tool)
    with qtbot.waitExposed(tool):
        tool.show()

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
