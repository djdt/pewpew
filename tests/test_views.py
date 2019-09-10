from pytestqt.qtbot import QtBot

from pewpew.widgets.views import ViewSpace, View, ViewTabBar, ViewTitleBar


def test_view_space(qtbot: QtBot):
    viewspace = ViewSpace()
    qtbot.addWidget(viewspace)
    assert len(viewspace.views) == 1

    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.splitHorizontal()
    assert len(viewspace.views) == 2

    viewspace.setActiveView(viewspace.views[1])
    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.splitVertical()
    assert len(viewspace.views) == 3

    viewspace.setActiveView(viewspace.views[2])
    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.splitHorizontal()
    assert len(viewspace.views) == 4

    # Should be 1 view on left, 1 on top right, 2 bottom right
    assert viewspace.count() == 2
    assert viewspace.widget(1).count() == 2
    assert viewspace.widget(1).widget(1).count() == 2

    with qtbot.wait_signal(viewspace.numViewsChanged):
        viewspace.closeView(viewspace.views[1])

    assert len(viewspace.views) == 3

    # Should be 1 view on left, 1 on top right, 1 bottom right
    assert viewspace.count() == 2
    assert viewspace.widget(1).count() == 2
    assert isinstance(viewspace.widget(1).widget(1), View)
