import numpy as np

from pytestqt.qtbot import QtBot

from pewlib.laser import Laser

from pewpew.widgets.laser import LaserViewSpace
from pewpew.widgets.tools.calculator import (
    CalculatorName,
    CalculatorFormula,
    CalculatorTool,
)
from testing import rand_data


def test_tool_calculator_name(qtbot: QtBot):
    lineedit = CalculatorName("a", ["b"], ["abs"])
    qtbot.addWidget(lineedit)

    assert lineedit.hasAcceptableInput()
    lineedit.setText("")
    assert not lineedit.hasAcceptableInput()
    lineedit.setText("a a")
    assert not lineedit.hasAcceptableInput()
    lineedit.setText("a\ta")
    assert not lineedit.hasAcceptableInput()
    lineedit.setText("a\na")
    assert not lineedit.hasAcceptableInput()
    lineedit.setText("b")
    assert not lineedit.hasAcceptableInput()
    lineedit.setText("abs")
    assert not lineedit.hasAcceptableInput()


def test_tool_calculator_formula(qtbot: QtBot):
    textedit = CalculatorFormula("", ["a", "b"])
    qtbot.addWidget(textedit)

    assert not textedit.hasAcceptableInput()
    textedit.setPlainText("a + 1")
    assert textedit.hasAcceptableInput()
    textedit.setPlainText("a + c")
    assert not textedit.hasAcceptableInput()


def test_tool_calculator(qtbot: QtBot):
    viewspace = LaserViewSpace()
    qtbot.addWidget(viewspace)
    viewspace.show()

    view = viewspace.activeView()
    widget = view.addLaser(Laser(rand_data(["a", "b"])))
    tool = CalculatorTool(view.activeWidget())
    view.addTab("Tool", tool)
    qtbot.waitForWindowShown(tool)

    assert tool.lineedit_name.text() == "calc0"
    tool.apply()
    assert "calc0" in widget.laser.elements
    assert tool.lineedit_name.text() == "calc1"

    # overwrite
    tool.lineedit_name.setText("a")
    tool.apply()
    assert len(widget.laser.elements) == 3

    # Inserters
    assert tool.formula.toPlainText() == "a"
    tool.insertFunction(1)
    assert tool.formula.toPlainText() == "abs(a"
    tool.insertVariable(2)
    assert tool.formula.toPlainText() == "abs(ba"

    # Test output of previewData and output lineedit
    x = np.array(np.random.random((10, 10)), dtype=[("a", float)])

    tool.formula.setPlainText("mean(a)")
    assert tool.previewData(x) is None
    assert tool.output.text() == f"{np.mean(x['a']):.10g}"

    # Array access in output
    tool.formula.setPlainText("a[0]")
    assert tool.previewData(x) is None
    assert tool.output.text() == f"{list(map('{:.4g}'.format, x['a'][0]))}"

    # Simple op
    tool.formula.setPlainText("a + 1.0")
    assert np.all(tool.previewData(x) == x["a"] + 1.0)
    assert tool.isComplete()

    # Invalid input
    tool.formula.setPlainText("fail")
    assert tool.previewData(x) is None
    assert not tool.isComplete()
