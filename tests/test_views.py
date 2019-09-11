from pytestqt.qtbot import QtBot

from PySide2 import QtCore, QtWidgets

from pewpew.widgets.views import ViewSpace, View, ViewTabBar, ViewTitleBar


def test_view_space_views(qtbot: QtBot):
    viewspace = ViewSpace()
    qtbot.addWidget(viewspace)
    assert len(viewspace.views) == 1

    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.splitActiveHorizontal()
    assert len(viewspace.views) == 2

    viewspace.setActiveView(viewspace.views[1])
    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.splitActiveVertical()
    assert len(viewspace.views) == 3

    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.splitView(viewspace.views[2], QtCore.Qt.Horizontal)
    assert len(viewspace.views) == 4

    # Should be 1 view on left, 1 on top right, 2 bottom right
    assert viewspace.count() == 2
    assert isinstance(viewspace.widget(0), View)
    assert isinstance(viewspace.widget(1), QtWidgets.QSplitter)
    assert viewspace.widget(1).count() == 2
    assert isinstance(viewspace.widget(1).widget(0), View)
    assert isinstance(viewspace.widget(1).widget(1), QtWidgets.QSplitter)
    assert viewspace.widget(1).widget(1).count() == 2
    assert isinstance(viewspace.widget(1).widget(1).widget(0), View)
    assert isinstance(viewspace.widget(1).widget(1).widget(1), View)

    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.closeView(viewspace.views[0])
    assert len(viewspace.views) == 3

    # Should be 1 view on top, 1 on bottom left, 1 bottom right
    # Original splitter changes orientation, inherits children of right splitter
    assert viewspace.count() == 2
    assert viewspace.orientation() == QtCore.Qt.Vertical
    assert isinstance(viewspace.widget(0), View)
    assert isinstance(viewspace.widget(1), QtWidgets.QSplitter)
    assert viewspace.widget(1).count() == 2
    assert isinstance(viewspace.widget(1).widget(0), View)
    assert isinstance(viewspace.widget(1).widget(1), View)

    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.closeActiveView()
    assert len(viewspace.views) == 2
    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.closeActiveView()
    assert len(viewspace.views) == 1
    with qtbot.assert_not_emitted(viewspace.numViewsChanged, wait=100):
        viewspace.closeActiveView()
    assert len(viewspace.views) == 1
