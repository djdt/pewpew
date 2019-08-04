from pytestqt.qtbot import QtBot

from pewpew.ui.docks.dockarea import DockArea


def test_dock_area(qtbot: QtBot):
    dockarea = DockArea()
    qtbot.addWidget(dockarea)
    dockarea.show()
