import numpy as np

from PySide2 import QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from pewlib.laser import Laser
from pewlib.process import convolve

from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools.tool import ToolWidget
from pewpew.widgets.tools.edit import (
    EditTool,
    CalculatorName,
    CalculatorFormula,
    CalculatorMethod,
)

from testing import linear_data, rand_data, FakeEvent


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


def test_edit_tool_calculator_name(qtbot: QtBot):
    lineedit = CalculatorName("a", ["b"], ["abs"])
    qtbot.addWidget(lineedit)

    assert lineedit.hasAcceptableInput()
    lineedit.setText("")
    assert not lineedit.hasAcceptableInput()
    lineedit.setText("b")
    assert not lineedit.hasAcceptableInput()
    lineedit.setText("abs")
    assert not lineedit.hasAcceptableInput()


def test_edit_tool_calculator_formula(qtbot: QtBot):
    textedit = CalculatorFormula("", ["a", "b"])
    qtbot.addWidget(textedit)

    assert not textedit.hasAcceptableInput()
    textedit.setPlainText("a + 1")
    assert textedit.hasAcceptableInput()
    textedit.setPlainText("a + c")
    assert not textedit.hasAcceptableInput()


def test_edit_tool_calculator(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    widget = view.addLaser(Laser(rand_data(["a", "b"])))
    tool = EditTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    tool.combo_method.setCurrentText("Calculator")
    assert not tool.combo_isotope.isEnabled()  # Full data tool

    assert tool.calculator_method.lineedit_name.text() == "calc0"
    tool.apply()
    assert "calc0" in widget.laser.isotopes
    assert tool.calculator_method.lineedit_name.text() == "calc1"

    # overwrite
    tool.calculator_method.lineedit_name.setText("a")
    tool.apply()
    assert len(widget.laser.isotopes) == 3

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

    # Array access in output
    tool.calculator_method.formula.setPlainText("a[0]")
    assert tool.calculator_method.previewData(x) is None
    assert (
        tool.calculator_method.output.text()
        == f"{list(map('{:.4g}'.format, x['a'][0]))}"
    )

    # Simple op
    tool.calculator_method.formula.setPlainText("a + 1.0")
    assert np.all(tool.calculator_method.previewData(x) == x["a"] + 1.0)
    assert tool.isComplete()

    # Invalid input
    tool.calculator_method.formula.setPlainText("fail")
    assert tool.calculator_method.previewData(x) is None
    assert not tool.isComplete()


def test_edit_tool_convolve(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    widget = view.addLaser(Laser(rand_data(["a"])))
    tool = EditTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    tool.combo_method.setCurrentText("Convolve")
    assert tool.combo_isotope.isEnabled()  # Single data tool

    tool.convolve_method.combo_kernel.setCurrentText("Gaussian")
    tool.convolve_method.combo_kernel.activated.emit(0)
    assert np.all(tool.convolve_method.kparams == [1.0, 0.0])

    # Change number of inputs
    tool.convolve_method.combo_kernel.setCurrentText("Exponential")
    tool.convolve_method.combo_kernel.activated.emit(0)
    assert np.all(tool.convolve_method.kparams == [1.0])
    tool.convolve_method.lineedit_kparams[0].setText("2.0")
    assert np.all(tool.convolve_method.kparams == [2.0])

    # Calid inputs stick around
    tool.convolve_method.combo_kernel.setCurrentText("Gaussian")
    tool.convolve_method.combo_kernel.activated.emit(0)
    assert np.all(tool.convolve_method.kparams == [2.0, 0.0])
    tool.convolve_method.lineedit_kparams[1].setText("-2.0")
    assert np.all(tool.convolve_method.kparams == [2.0, -2.0])

    # Bad inputs go to defaults
    tool.convolve_method.combo_kernel.setCurrentText("Triangular")
    tool.convolve_method.combo_kernel.activated.emit(0)
    assert np.all(tool.convolve_method.kparams == [-2.0, 2.0])

    # Test isComplete
    assert tool.convolve_method.isComplete()
    tool.convolve_method.lineedit_kparams[0].setText("4.0")
    assert not tool.convolve_method.isComplete()
    tool.convolve_method.lineedit_kparams[0].setText("-2.0")
    assert tool.convolve_method.isComplete()
    tool.convolve_method.lineedit_kscale.setText("-1.0")
    assert not tool.convolve_method.isComplete()
    tool.convolve_method.lineedit_kscale.setText("1.0")
    assert tool.convolve_method.isComplete()
    tool.convolve_method.lineedit_ksize.setText("1")
    assert not tool.convolve_method.isComplete()

    # Tset convolve is correct
    tool.convolve_method.combo_kernel.setCurrentText("Gaussian")
    tool.convolve_method.lineedit_ksize.setText("3")
    tool.convolve_method.lineedit_kparams[0].setText("1.0")
    tool.convolve_method.lineedit_kparams[1].setText("0.0")
    tool.convolve_method.combo_kernel.activated.emit(0)

    x = np.random.random((10, 10))
    psf = convolve.normal(3, 1.0, 0.0, scale=1.0)[:, 1]

    tool.convolve_method.combo_horizontal.setCurrentText("Right to Left")
    tool.convolve_method.combo_vertical.setCurrentText("Bottom to Top")

    c = np.apply_along_axis(convolve.convolve, 1, x[::-1, ::-1], psf, mode="pad")
    c = np.apply_along_axis(convolve.convolve, 0, c, psf, mode="pad")
    c = c[::-1, ::-1]

    assert np.all(tool.convolve_method.previewData(x) == c)
    tool.apply()


def test_edit_tool_deconvolve(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    widget = view.addLaser(Laser(rand_data(["a"])))
    tool = EditTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    tool.combo_method.setCurrentText("Deconvolve")
    assert tool.combo_isotope.isEnabled()  # Single data tool

    tool.deconvolve_method.combo_kernel.setCurrentText("Gaussian")
    tool.deconvolve_method.lineedit_ksize.setText("3")
    tool.deconvolve_method.lineedit_kparams[0].setText("1.0")
    tool.deconvolve_method.lineedit_kparams[1].setText("0.0")
    tool.deconvolve_method.combo_kernel.activated.emit(0)

    x = np.random.random((10, 10))
    psf = convolve.normal(3, 1.0, 0.0, scale=1.0)[:, 1]

    tool.deconvolve_method.combo_horizontal.setCurrentText("Right to Left")
    tool.deconvolve_method.combo_vertical.setCurrentText("Bottom to Top")

    c = np.apply_along_axis(convolve.deconvolve, 1, x[::-1, ::-1], psf, mode="same")
    c = np.apply_along_axis(convolve.deconvolve, 0, c, psf, mode="same")
    c = c[::-1, ::-1]

    assert np.all(tool.deconvolve_method.previewData(x) == c)
    tool.apply()


def test_edit_tool_filter(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()
    view = viewspace.activeView()
    widget = view.addLaser(Laser(rand_data(["a"])))
    tool = EditTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    tool.combo_method.setCurrentText("Filter")
    assert tool.combo_isotope.isEnabled()  # Single data tool


    tool.filter_method.combo_filter.setCurrentText("Mean")
    tool.filter_method.combo_filter.activated.emit(0)
    tool.filter_method.lineedit_fsize.setText("3")
    tool.filter_method.lineedit_fparams[0].setText("3.0")
    tool.filter_method.lineedit_fparams[0].editingFinished.emit()
    assert tool.filter_method.fsize == 3
    assert np.all(tool.filter_method.fparams == [3.0])
    assert tool.isComplete()

    tool.filter_method.lineedit_fparams[0].setText("2.0")
    tool.filter_method.lineedit_fparams[0].editingFinished.emit()
    assert np.all(tool.filter_method.fparams == [2.0])
    assert tool.isComplete()

    tool.filter_method.combo_filter.setCurrentText("Median")
    tool.filter_method.combo_filter.activated.emit(0)
    assert np.all(tool.filter_method.fparams == [2.0])
    assert tool.isComplete()

    tool.filter_method.lineedit_fparams[0].setText("-1.0")
    tool.filter_method.lineedit_fparams[0].editingFinished.emit()
    assert not tool.isComplete()

    tool.filter_method.combo_filter.setCurrentText("Mean")
    tool.filter_method.combo_filter.activated.emit(0)
    assert np.all(tool.filter_method.fparams == [3.0])
    assert tool.isComplete()

    tool.apply()
