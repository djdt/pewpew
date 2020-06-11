import numpy as np
import os.path
import tempfile

from PySide2 import QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from pew.laser import Laser

from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools.tool import ToolWidget
from pewpew.widgets.tools.edit import EditTool
from pewpew.widgets.tools.standards import StandardsTool
from pewpew.widgets.tools.overlays import OverlayTool

from testing import linear_data, rand_data, FakeEvent


def test_tool_widget(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    view.addLaser(Laser(rand_data("A1")))
    tool = ToolWidget(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

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
        == "RSQ\t1.0000\nGradient\t2.0000\nIntercept\t0.5000\nSxy\t0.0000\nLOD (3σ)\t0.0000"
    )

    dlg = tool.showCurve()
    qtbot.waitForWindowShown(dlg)
    dlg.close()


def test_edit_tool(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    widget = view.addLaser(Laser(rand_data(["a", "b"])))
    tool = EditTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    # Transform tools
    tool.actionTransformFlipHorz()
    assert tool.flip_horizontal
    tool.actionTransformFlipHorz()
    tool.actionTransformFlipVert()
    assert tool.flip_vertical
    tool.actionTransformFlipVert()
    tool.actionTransformRotateLeft()
    assert tool.rotate == 3
    tool.actionTransformRotateRight()
    assert tool.rotate == 0

    assert tool.combo_method.currentText() == "Calculator"
    assert not tool.combo_isotope.isEnabled()  # Full data tool

    assert tool.calculator_method.lineedit_name.text() == "calc0"
    tool.calculator_method.apply()
    # Applying should add isotope to widget
    isotopes = [
        tool.widget.combo_isotope.itemText(i)
        for i in range(widget.combo_isotope.count())
    ]
    assert "calc0" in isotopes
    isotopes = [
        widget.combo_isotope.itemText(i) for i in range(widget.combo_isotope.count())
    ]
    assert "calc0" in isotopes

    # Inserters
    assert tool.calculator_method.formula.toPlainText() == "a"
    tool.calculator_method.insertFunction(1)
    assert tool.calculator_method.formula.toPlainText() == "abs(a"
    tool.calculator_method.insertVariable(2)
    assert tool.calculator_method.formula.toPlainText() == "abs(ba"

    # Test output of previewData and output lineedit
    x = np.array(np.random.random((10, 10)), dtype=[("a", float)])

    tool.calculator_method.formula.setPlainText("mean(a)")
    assert tool.calculator_method.previewData(x) is None
    assert tool.calculator_method.output.text() == f"{np.mean(x['a']):.10g}"

    tool.calculator_method.formula.setPlainText("a[0]")
    assert tool.calculator_method.previewData(x) is None
    assert (
        tool.calculator_method.output.text()
        == f"{list(map('{:.4g}'.format, x['a'][0]))}"
    )

    tool.calculator_method.formula.setPlainText("a + 1.0")
    assert np.all(tool.calculator_method.previewData(x) == x["a"] + 1.0)

    tool.calculator_method.formula.setPlainText("fail")
    assert tool.calculator_method.previewData(x) is None

    tool.combo_method.setCurrentText("Convolve")
    assert tool.combo_isotope.isEnabled()
    tool.convolve_method.apply()
    tool.combo_method.setCurrentText("Deconvolve")
    assert tool.combo_isotope.isEnabled()
    tool.deconvolve_method.apply()
    tool.combo_method.setCurrentText("Filter")
    assert tool.combo_isotope.isEnabled()
    tool.filter_method.apply()
    tool.combo_method.setCurrentText("Transform")
    assert tool.combo_isotope.isEnabled()
    tool.transform_method.apply()


# def test_calculations_tool(qtbot: QtBot):
#     viewspace = LaserViewSpace()
#     qtbot.addWidget(viewspace)
#     viewspace.show()
#     view = viewspace.activeView()
#     view.addLaser(Laser(rand_data("A1")))
#     tool = CalculationsTool(view.activeWidget())
#     view.addTab("Tool", tool)
#     qtbot.waitForWindowShown(tool)

#     assert not tool.isComplete()

#     tool.formula.setText("1 +")
#     assert tool.formula.expr == ""
#     tool.combo_isotope.setCurrentIndex(1)
#     tool.insertVariable(1)
#     assert tool.formula.expr == "+ 1 A1"
#     assert tool.combo_isotope.currentIndex() == 0

#     assert not tool.isComplete()

#     tool.lineedit_name.setText("A1")
#     assert not tool.lineedit_name.hasAcceptableInput()
#     tool.lineedit_name.setText("A2 ")
#     assert not tool.lineedit_name.hasAcceptableInput()
#     tool.lineedit_name.setText("if")
#     assert not tool.lineedit_name.hasAcceptableInput()
#     tool.lineedit_name.setText(" ")
#     assert not tool.lineedit_name.hasAcceptableInput()
#     tool.lineedit_name.setText("A2")
#     assert tool.lineedit_name.hasAcceptableInput()

#     assert tool.isComplete()

#     tool.apply()
#     assert np.all(tool.widget.laser.data["A2"] == tool.widget.laser.data["A1"] + 1.0)


def test_overlay_tool(qtbot: QtBot):
    data = np.zeros((10, 10), dtype=[("r", float), ("g", float), ("b", float)])
    data["r"][:, :] = 1.0
    data["g"][:10, :] = 1.0
    data["b"][:, :10] = 1.0

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
    tool.comboAdd(1)  # r
    assert np.all(tool.canvas.image.get_array() == (1.0, 0.0, 0.0))

    tool.comboAdd(2)  # g
    assert np.all(tool.canvas.image.get_array()[:10] == (1.0, 1.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[10:] == (1.0, 0.0, 0.0))

    tool.comboAdd(3)  # g
    assert np.all(tool.canvas.image.get_array()[:10, :10] == (1.0, 1.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[10:, :10] == (1.0, 0.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[10:, 10:] == (1.0, 1.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[10:, 10:] == (1.0, 0.0, 0.0))

    # Test cmyk mode
    tool.radio_cmyk.toggle()
    assert tool.rows.color_model == "cmyk"
    assert np.all(tool.canvas.image.get_array()[:10, :10] == (0.0, 0.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[10:, :10] == (0.0, 1.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[10:, 10:] == (0.0, 0.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[10:, 10:] == (0.0, 1.0, 1.0))

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
    tool.addRow("g")
    tool.rows[3].setColor(QtGui.QColor.fromRgbF(0.0, 1.0, 1.0))
    assert tool.rows.rowCount() == 4

    # Test normalise
    assert np.amin(tool.canvas.image.get_array()) > 0.0
    tool.check_normalise.setChecked(True)
    tool.refresh()
    assert tool.canvas.image.get_array().data.min() == 0.0
    tool.check_normalise.setChecked(False)

    # Test export
    dlg = tool.openExportDialog()

    dlg2 = dlg.selectDirectory()
    dlg2.close()

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
    assert np.all(tool.canvas.image.get_array()[:10, :10] == (0.0, 1.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[10:, :10] == (0.0, 0.0, 1.0))
    assert np.all(tool.canvas.image.get_array()[10:, 10:] == (0.0, 1.0, 0.0))
    assert np.all(tool.canvas.image.get_array()[10:, 10:] == (0.0, 0.0, 0.0))

    dlg = tool.rows[0].selectColor()
    dlg.close()


# def test_tools_main_window(qtbot: QtBot):
#     window = MainWindow()
#     qtbot.addWidget(window)
#     window.show()

#     window.viewspace.views[0].addLaser(Laser(rand_data("A1")))
#     window.viewspace.views[0].addTab(
#         "Tool 1", StandardsTool(window.viewspace.activeWidget())
#     )
#     window.viewspace.views[0].addTab(
#         "Tool 2", EditTool(window.viewspace.activeWidget())
#     )
#     window.viewspace.views[0].addTab(
#         "Tool 3", OverlayTool(window.viewspace.activeWidget())
#     )
#     window.viewspace.refresh()

#     window.actionToggleCalibrate(False)
#     window.actionToggleColorbar(False)
#     window.actionToggleLabel(False)
#     window.actionToggleScalebar(False)
