from pytestqt.qtbot import QtBot

from pewpew.widgets.tools import Tool, StandardsTool, OperationsTool


def test_tool(qtbot: QtBot):
    tool = Tool()
    qtbot.addWidget(tool)
    tool.show()


# def test_standards_tool(qtbot: QtBot):
#     tool = StandardsTool()
#     qtbot.addWidget(tool)
#     tool.show()


# def test_operations_tool(qtbot: QtBot):
#     tool = OperationsTool()
#     qtbot.addWidget(tool)
#     tool.show()
