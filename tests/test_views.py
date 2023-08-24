from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from pewpew.widgets.views import TabView, TabViewBar, TabViewWidget

# Most of the drag/drop actions are not tested and have no cover.


class TestTabViewWidget(TabViewWidget):
    def __init__(self, idx: int, view: TabView, editable: bool = False):
        self.idx = idx
        super().__init__(view, editable)

    def rename(self, text: str) -> None:
        pass


def test_tab_view_active(qtbot: QtBot):
    view = TabView()
    qtbot.addWidget(view)
    view.show()
    # Default active
    assert view.activeWidget() is None

    widget = TestTabViewWidget(0, view)
    view.addTab("0", widget)
    # Focus new widget
    assert view.activeWidget() == widget

    # only focus if no existing tab
    widget1 = TestTabViewWidget(1, view)
    view.addTab("1", widget1)
    assert view.activeWidget() == widget
    view.setActiveWidget(widget1)
    assert view.activeWidget() == widget1

    # Test focus on tab changes active
    qtbot.mouseClick(view.tabs, QtCore.Qt.LeftButton, pos=view.tabs.tabRect(0).center())
    assert view.activeWidget() == widget


def test_tab_view_add_remove(qtbot: QtBot):
    view = TabView()
    qtbot.addWidget(view)
    view.show()

    assert len(view.widgets()) == 0
    view.addTab("0", TestTabViewWidget(0, view))
    view.addTab("1", TestTabViewWidget(1, view))
    view.addTab("2", TestTabViewWidget(2, view))
    assert len(view.widgets()) == 3
    view.removeTab(0)
    assert len(view.widgets()) == 2


# def test_view_tabs(qtbot: QtBot):
#     view = view()
#     qtbot.addWidget(view)
#     view.show()
#     view = view.activeView()
#     # Creating tabs
#     with qtbot.waitSignal(view.numTabsChanged):
#         view.addTab("1", _TestViewWidget(1, view))
#     with qtbot.waitSignal(view.numTabsChanged):
#         view.addTab("3", _TestViewWidget(3, view))
#     with qtbot.waitSignal(view.numTabsChanged):
#         view.insertTab(1, "2", _TestViewWidget(2, view))
#     assert view.tabs.count() == 3
#     assert [view.tabs.tabText(i) for i in range(3)] == ["1", "2", "3"]
#     assert [view.stack.widget(i).idx for i in range(3)] == [1, 2, 3]
#     # Moving tabs
#     with qtbot.assertNotEmitted(view.numTabsChanged, wait=100):
#         view.tabs.moveTab(2, 0)
#     assert [view.tabs.tabText(i) for i in range(3)] == ["3", "1", "2"]
#     assert [view.stack.widget(i).idx for i in range(3)] == [3, 1, 2]
#     # Removing tabs
#     with qtbot.waitSignal(view.numTabsChanged):
#         view.removeTab(1)
#     assert view.tabs.count() == 2
#     assert [view.tabs.tabText(i) for i in range(2)] == ["3", "2"]
#     assert [view.stack.widget(i).idx for i in range(2)] == [3, 2]

#     # view.setTabModified(0, True)
#     # assert view.tabs.tabIcon(0).name() == "document-save"
#     # assert view.tabs.tabIcon(1).name() == ""

#     view.removeTab(0)
#     assert len(view.widgets()) == 1
#     view.removeTab(0)
#     assert len(view.widgets()) == 0


# def test_view_tab_bar(qtbot: QtBot):
#     view = view()
#     qtbot.addWidget(view)
#     view.show()
#     view.splitActiveHorizontal()
#     view = view.views[0]
#     tabs = view.tabs

#     tabs.view.addTab("1", _TestViewWidget(1, view, editable=True))
#     tabs.view.addTab("2", _TestViewWidget(2, view))
#     # Test double click rename
#     dlg = tabs.tabRenameDialog(0)
#     assert dlg.textValue() == "1"
#     dlg.textValueSelected.emit("3")
#     dlg.close()
#     assert tabs.tabText(0) == "3"
#     # Rename on non editable will not open dialog
#     assert tabs.tabRenameDialog(1) is None

#     # Test drag and drop same bar
#     with qtbot.assertNotEmitted(tabs.view.numTabsChanged, wait=100):
#         qtbot.mousePress(tabs, QtCore.Qt.LeftButton, pos=tabs.tabRect(0).center())
#         qtbot.mouseRelease(tabs, QtCore.Qt.LeftButton, pos=tabs.tabRect(1).center())
#     assert tabs.tabText(0) == "3"
#     assert tabs.tabText(1) == "2"
#     # Test drag and drop to new bar / view
#     # Broken in QTest


# # def test_view_title_bar(qtbot: QtBot):
# #     view = view()
# #     qtbot.addWidget(view)
# #     view.show()
# #     view.splitActiveHorizontal()

# #     # Blocking
# #     qtbot.mouseClick(view.views[1].titlebar.split_button, QtCore.Qt.LeftButton)
# #     view.view
# #     assert view.views[1].active
