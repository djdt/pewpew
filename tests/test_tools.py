import numpy as np
from PySide2 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from laserlib.laser import Laser

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.tools import Tool, StandardsTool, CalculationsTool


def test_tool(qtbot: QtBot):
    tool = Tool()
    qtbot.addWidget(tool)
    tool.show()

    with qtbot.waitSignal(tool.mouseSelectStarted):
        tool.startMouseSelect()
        assert not tool.isVisible()

    with qtbot.waitSignal(tool.mouseSelectEnded):
        tool.endMouseSelect()
        assert tool.isVisible()

    qtbot.keyPress(tool, QtCore.Qt.Key_Escape)
    assert tool.isVisible()
    tool.close()


def test_standards_tool(qtbot: QtBot):
    laser = Laser.from_structured(
        np.array(np.arange(100).reshape((10, 10)), dtype=[("A1", float), ("B2", float)])
    )
    viewoptions = ViewOptions()
    tool = StandardsTool(LaserWidget(laser, viewoptions, None))
    qtbot.addWidget(tool)
    tool.show()

    tool.lineedit_units.setText("unit")
    tool.lineedit_units.editingFinished.emit()
    tool.combo_weighting.setCurrentIndex(1)

    tool.combo_trim.setCurrentIndex(0)
    tool.lineedit_left.setText("2")
    tool.lineedit_left.editingFinished.emit()
    assert tool.canvas.view_limits == (70.0, 350.0, 0.0, 350.0)

    tool.lineedit_right.setText("2")
    tool.lineedit_right.editingFinished.emit()
    assert tool.canvas.view_limits == (70.0, 280.0, 0.0, 350.0)

    tool.spinbox_levels.setValue(5)

    assert not tool.isComplete()
    assert not tool.results_box.button.isEnabled()
    for i in range(0, tool.table.model().rowCount()):
        index = tool.table.model().index(i, 0)
        tool.table.model().setData(index, 0)
    assert not tool.table.isComplete()
    for i in range(0, tool.table.model().rowCount()):
        index = tool.table.model().index(i, 0)
        tool.table.model().setData(index, i)
    assert tool.isComplete()
    assert tool.results_box.button.isEnabled()

    # Change isotope, check if weighting and unit have remained
    tool.combo_isotope.setCurrentIndex(1)
    assert not tool.isComplete()
    assert tool.combo_weighting.currentIndex() == 1
    assert tool.lineedit_units.text() == "unit"
    tool.lineedit_units.setText("none")
    tool.combo_weighting.setCurrentIndex(2)

    # Change isotope back, check weighting, unit, table restored
    tool.combo_isotope.setCurrentIndex(0)
    assert tool.isComplete()
    assert tool.lineedit_units.text() == "unit"
    assert tool.combo_weighting.currentIndex() == 1

    tool.results_box.copy()
    assert (
        QtWidgets.QApplication.clipboard().text()
        == "RSQ\t1.0000\nGradient\t20.0000\nIntercept\t9.5000"
    )

    dlg = tool.showCurve()
    dlg.close()


def test_calculations_tool(qtbot: QtBot):
    laser = Laser.from_structured(
        np.array(np.random.random((10, 10)), dtype=[("A1", float)])
    )
    viewoptions = ViewOptions()
    tool = CalculationsTool(LaserWidget(laser, viewoptions, None))
    qtbot.addWidget(tool)
    tool.show()

    assert not tool.isComplete()

    tool.formula.setText("1 +")
    assert tool.formula.expr == ""
    tool.combo_isotopes.setCurrentIndex(1)
    tool.insertVariable(1)
    assert tool.formula.expr == "+ 1 A1"
    assert tool.combo_isotopes.currentIndex() == 0

    assert not tool.isComplete()

    tool.lineedit_name.setText("A1")
    assert not tool.lineedit_name.hasAcceptableInput()
    tool.lineedit_name.setText("A2 ")
    assert not tool.lineedit_name.hasAcceptableInput()
    tool.lineedit_name.setText("if")
    assert not tool.lineedit_name.hasAcceptableInput()
    tool.lineedit_name.setText(" ")
    assert not tool.lineedit_name.hasAcceptableInput()
    tool.lineedit_name.setText("A2")
    assert tool.lineedit_name.hasAcceptableInput()

    assert tool.isComplete()

    tool.apply()
    assert np.all(laser.data["A2"].data == laser.data["A1"].data + 1.0)
