from pathlib import Path

import numpy as np
from pewlib.laser import Laser
from pytestqt.qtbot import QtBot
from testing import rand_data

from pewpew.widgets.laser import LaserTabView
from pewpew.widgets.tools.calculator import (
    CalculatorFormula,
    CalculatorName,
    CalculatorTool,
)


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
    view = LaserTabView()
    qtbot.addWidget(view)
    view.show()

    widget = view.importFile(
        Path("/home/pewpew/fake.npz"),
        Laser(rand_data(["a", "b"]), info={"Name": "test"}),
    )
    item = widget.laserItems()[0]
    tool = CalculatorTool(item)
    view.addTab("Tool", tool)
    with qtbot.waitExposed(tool):
        tool.show()

    assert tool.lineedit_name.text() == "calc0"
    tool.apply()
    assert "calc0" in item.laser.elements
    assert tool.lineedit_name.text() == "calc1"

    # overwrite
    tool.lineedit_name.setText("a")
    tool.apply()
    assert len(item.laser.elements) == 3

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
