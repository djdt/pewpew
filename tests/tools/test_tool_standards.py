import numpy as np
from pewlib.laser import Laser
from pytestqt.qtbot import QtBot
from testing import linear_data

from pewpew.widgets.laser import LaserTabView
from pewpew.widgets.tools.standards import StandardsTool


def test_standards_tool(qtbot: QtBot):
    data = linear_data(["A1", "B2"])
    data["B2"][6] = 1.0
    view = LaserTabView()
    qtbot.add_widget(view)
    view.show()
    widget = view.importFile(
        Laser(data, info={"Name": "test", "File Path": "/home/pewpew/real.npz"})
    )
    item = widget.laserItems()[0]
    tool = StandardsTool(item)
    view.addTab("Tool", tool)
    with qtbot.waitExposed(tool):
        tool.show()

    # Units
    tool.lineedit_units.setText("unit")
    tool.lineedit_units.editingFinished.emit()
    tool.combo_weighting.setCurrentIndex(2)

    # Table
    tool.spinbox_levels.setValue(5)

    assert not tool.isComplete()
    assert not tool.button_plot.isEnabled()
    for i in range(0, tool.table.model().rowCount()):
        index = tool.table.model().index(i, 0)
        tool.table.model().setData(index, i)

    # Change element, check weighting
    tool.combo_element.setCurrentIndex(1)
    assert not tool.table.isComplete()
    for i in range(0, tool.table.model().rowCount()):
        index = tool.table.model().index(i, 0)
        tool.table.model().setData(index, i)
    assert tool.isComplete()
    assert tool.button_plot.isEnabled()

    assert tool.combo_weighting.currentIndex() == 0
    assert tool.lineedit_units.text() == ""
    tool.lineedit_units.setText("none")
    assert tool.calibration["B2"].gradient == 1.75

    # Check weighting updates results
    tool.combo_weighting.setCurrentText("y")
    assert tool.calibration["B2"].weighting == "y"
    assert np.isclose(tool.calibration["B2"].gradient, 1.954022988)

    # Change element back, check weighting, unit, table restored
    tool.combo_element.setCurrentIndex(0)
    assert tool.isComplete()
    assert tool.lineedit_units.text() == "unit"
    assert tool.combo_weighting.currentIndex() == 2

    # Test SD weighting
    tool.combo_weighting.setCurrentIndex(1)
    assert np.all(tool.calibration["A1"].weights == 4.0)

    tool.combo_element.setCurrentIndex(1)
    dlg = tool.showCurve()
    qtbot.waitExposed(dlg)
    dlg.close()
