import numpy as np
import os.path
import tempfile

from PySide2 import QtWidgets
from pytestqt.qtbot import QtBot

from pew.laser import Laser

from pewpew.mainwindow import MainWindow
from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools import (
    ToolWidget,
    StandardsTool,
    CalculationsTool,
    OverlayTool,
)

from testing import linear_data, rand_data, FakeEvent


def test_tool_widget(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    view.addLaser(Laser(rand_data("A1")))
    tool = ToolWidget(view.activeWidget())
    view.addTab("Tool", tool)

    tool.startMouseSelect()
    assert view.activeWidget() != tool
    tool.endMouseSelect()
    assert view.activeWidget() == tool


def test_standards_tool(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    view.addLaser(Laser(linear_data(["A1", "B2"])))
    tool = StandardsTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    # Units
    tool.lineedit_units.setText("unit")
    tool.lineedit_units.editingFinished.emit()
    tool.combo_weighting.setCurrentIndex(2)

    # Trim
    assert tool.canvas.getCurrentTrim() == (1, 9)

    tool.canvas.picked_artist = tool.canvas.v_guides[0]
    tool.canvas.move(FakeEvent(tool.canvas.ax, 90, 30))
    tool.canvas.release(FakeEvent(tool.canvas.ax, 90, 30))

    assert tool.canvas.getCurrentTrim() == (3, 9)

    # Test snap
    tool.canvas.picked_artist = tool.canvas.v_guides[0]
    tool.canvas.move(FakeEvent(tool.canvas.ax, 34, 30))
    tool.canvas.release(FakeEvent(tool.canvas.ax, 34, 30))

    assert tool.canvas.getCurrentTrim() == (1, 9)

    # Table
    tool.spinbox_levels.setValue(5)

    assert not tool.isComplete()
    assert not tool.results_box.button_plot.isEnabled()
    for i in range(0, tool.table.model().rowCount()):
        index = tool.table.model().index(i, 0)
        tool.table.model().setData(index, 0)
    assert not tool.table.isComplete()
    for i in range(0, tool.table.model().rowCount()):
        index = tool.table.model().index(i, 0)
        tool.table.model().setData(index, i)
    assert tool.isComplete()
    assert tool.results_box.button_plot.isEnabled()

    # Change isotope, check if weighting and unit have remained
    tool.combo_isotope.setCurrentIndex(1)
    assert not tool.isComplete()
    assert tool.combo_weighting.currentIndex() == 0
    assert tool.lineedit_units.text() == "unit"
    tool.lineedit_units.setText("none")
    tool.combo_weighting.setCurrentIndex(2)

    # Change isotope back, check weighting, unit, table restored
    tool.combo_isotope.setCurrentIndex(0)
    assert tool.isComplete()
    assert tool.lineedit_units.text() == "unit"
    assert tool.combo_weighting.currentIndex() == 2

    tool.results_box.copy()
    assert (
        QtWidgets.QApplication.clipboard().text()
        == "RSQ\t1.0000\nGradient\t2.0000\nIntercept\t0.5000\nY-error\t0.0000"
    )

    dlg = tool.showCurve()
    qtbot.waitForWindowShown(dlg)
    dlg.close()


def test_calculations_tool(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    view.addLaser(Laser(rand_data("A1")))
    tool = CalculationsTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

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
    assert np.all(tool.widget.laser.data["A2"] == tool.widget.laser.data["A1"] + 1.0)


def test_overlay_tool(qtbot: QtBot):
    data = np.zeros((20, 20), dtype=[("r", float), ("g", float), ("b", float)])
    data["r"][:, :] = 1.0
    data["g"][:20, :] = 1.0
    data["b"][:, :20] = 1.0

    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    view.addLaser(Laser(data))
    tool = OverlayTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    # Test rgb mode
    assert tool.rows.color_model == "rgb"
    tool.addRow("r")
    assert np.all(tool.canvas.image.get_array() == (1.0, 0.0, 0.0))
    tool.addRow("g")
    assert np.all(tool.canvas.image.get_array()[:20] == (1.0, 1.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[20:] == (1.0, 0.0, 0.0))
    tool.addRow("b")
    assert np.all(tool.canvas.image.get_array()[:20, :20] == (1.0, 1.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[20:, :20] == (1.0, 0.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[20:, 20:] == (1.0, 1.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[20:, 20:] == (1.0, 0.0, 0.0))

    # Test cmyk mode
    tool.radio_cmyk.toggle()
    assert tool.rows.color_model == "cmyk"
    assert np.all(tool.canvas.image.get_array()[:20, :20] == (0.0, 0.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[20:, :20] == (0.0, 1.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[20:, 20:] == (0.0, 0.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[20:, 20:] == (0.0, 1.0, 1.0))

    # Check that the rows are limited to 3
    assert tool.rows.max_rows == 3
    assert not tool.combo_add.isEnabled()
    assert tool.rows.rowCount() == 3
    with qtbot.assert_not_emitted(tool.rows.rowsChanged):
        tool.addRow("r")
    assert tool.rows.rowCount() == 3

    # Check color buttons are not enabled
    for row in tool.rows.rows:
        assert not row.button_color.isEnabled()

    # Test any mode
    tool.radio_custom.toggle()
    assert tool.rows.color_model == "any"
    assert tool.combo_add.isEnabled()
    for row in tool.rows.rows:
        assert row.button_color.isEnabled()
    tool.addRow("r")
    assert tool.rows.rowCount() == 4

    # Test export
    dlg = tool.openExportDialog()

    with tempfile.NamedTemporaryFile() as tf:
        dlg.export(tf)
        assert os.path.exists(tf.name)

    with tempfile.TemporaryDirectory() as td:
        dlg.lineedit_directory.setText(td)
        dlg.lineedit_filename.setText("test.png")
        dlg.check_individual.setChecked(True)
        dlg.accept()
        qtbot.wait(300)
        assert os.path.exists(os.path.join(td, "test_1.png"))
        assert os.path.exists(os.path.join(td, "test_2.png"))
        assert os.path.exists(os.path.join(td, "test_3.png"))

    # Test close
    with qtbot.wait_signal(tool.rows.rowsChanged):
        tool.rows.rows[-1].close()
    assert tool.rows.rowCount() == 3

    # Test hide
    tool.radio_rgb.toggle()
    with qtbot.wait_signal(tool.rows.rows[0].itemChanged):
        tool.rows.rows[0].button_hide.click()
    assert np.all(tool.canvas.image.get_array()[:20, :20] == (0.0, 1.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[20:, :20] == (0.0, 0.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[20:, 20:] == (0.0, 1.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[20:, 20:] == (0.0, 0.0, 0.0))


def test_tools_main_window(qtbot: QtBot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    window.viewspace.views[0].addLaser(Laser(rand_data("A1")))
    window.viewspace.views[0].addTab(
        "Tool 1", StandardsTool(window.viewspace.activeWidget())
    )
    window.viewspace.views[0].addTab(
        "Tool 2", CalculationsTool(window.viewspace.activeWidget())
    )
    window.viewspace.views[0].addTab(
        "Tool 3", OverlayTool(window.viewspace.activeWidget())
    )
    window.viewspace.refresh()

    window.actionToggleCalibrate(False)
    window.actionToggleColorbar(False)
    window.actionToggleLabel(False)
    window.actionToggleScalebar(False)
