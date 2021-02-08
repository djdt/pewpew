import numpy as np

from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools.standards import StandardsTool

from testing import linear_data


def test_standards_tool(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    data = linear_data(["A1", "B2"])
    data["B2"][6] = 1.0
    view.addLaser(Laser(data))
    tool = StandardsTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

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

    # Change isotope, check weighting
    tool.combo_isotope.setCurrentIndex(1)
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

    # Change isotope back, check weighting, unit, table restored
    tool.combo_isotope.setCurrentIndex(0)
    assert tool.isComplete()
    assert tool.lineedit_units.text() == "unit"
    assert tool.combo_weighting.currentIndex() == 2

    # Test SD weighting
    tool.combo_weighting.setCurrentIndex(1)
    assert np.all(tool.calibration["A1"].weights == 4.0)

    tool.combo_isotope.setCurrentIndex(1)
    dlg = tool.showCurve()
    qtbot.waitForWindowShown(dlg)
    dlg.close()
