from PySide2 import QtCore, QtGui, QtWidgets


class View:
    pass


class ViewSpace(QtWidgets.QSplitter):
    view_index = 0

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(orientation, parent)
        self.active_view = None

        self.action_split_horz = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-split-left-right"), "Split &Horizontal"
        )
        self.action_split_horz.triggered.connect(self.slotSplitHorizontal)
        self.action_split_vert = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-split-top-bottom"), "Split &Vertical"
        )
        self.action_split_vert.triggered.connect(self.slotSplitVertical)
        self.action_close_view = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-close"), "Close View"
        )
        self.action_close_view.triggered.connect(self.closeView)

    def slotSplitHorizontal(self) -> None:
        self.splitView()

    def slotSplitVertical(self) -> None:
        self.splitView(None, QtCore.Qt.Vertical)

    def closeView(self) -> None:
        if self.active_view is not None and self.count() > 1:
            self.active_view.close()
            self.active_view = None

    def splitView(
        self,
        view: QtWidgets.QWidget = None,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
    ) -> None:
        if view is None:
            view = self.activeView()

        splitter = view.parent()
        index = splitter.indexOf(view)

        if splitter.count() == 1:
            size = (splitter.sizes()[0] - splitter.handleWidth()) / 2.0
            splitter.setOrientation(orientation)
            splitter.insertWidget(index - 1, View(self, splitter))
            splitter.setSizes([size, size])
        else:
            sizes = splitter.sizes()
            new_splitter = QtWidgets.QSplitter(orientation)
            new_splitter.setChildrenCollapsible(False)
            new_splitter.addWidget(view)
            new_splitter.addWidget(View(self, new_splitter))
            splitter.insertWidget(index, new_splitter)

            splitter.setSizes(sizes)

            new_size = (sum(new_splitter.sizes()) - new_splitter.handleWidth()) / 2.0
            new_splitter.setSizes([new_size, new_size])

    def activeView(self) -> View:
        if self.active_view is None:
            view = self.findChildren(View)[0]
            self.active_view = view
        return self.active_view

    def setActiveView(self, view: View) -> None:
        self.active_view = view


class View(QtWidgets.QWidget):
    def __init__(self, viewspace: ViewSpace, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.viewspace = viewspace

        self.stack = QtWidgets.QStackedWidget()
        self.titlebar = ViewTitleBar(self)

        box = QtWidgets.QGroupBox()
        layout_box = QtWidgets.QVBoxLayout()
        layout_box.setSpacing(0)
        layout_box.setContentsMargins(0, 0, 0, 0)
        layout_box.addWidget(self.stack)
        box.setLayout(layout_box)

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.titlebar, 0)
        layout.addWidget(box, 1)
        self.setLayout(layout)

    def addTab(self, widget: QtWidgets.QWidget, title: str) -> None:
        index = self.titlebar.tabs.addTab(title)
        self.stack.insertWidget(index, widget)

    def changeTab(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    def removeTab(self, index: int) -> None:
        w = self.stack.widget(index)
        if w is not None:
            self.stack.removeWidget(w)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj and event.type() == QtCore.QEvent.MouseButtonPress:
            self.viewspace.setActiveView(self)
        return False


class ViewTitleBar(QtWidgets.QWidget):
    def __init__(self, view: View):
        super().__init__(view)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.view = view

        self.tabs = QtWidgets.QTabBar()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.tabs.removeTab)
        self.tabs.tabCloseRequested.connect(self.view.removeTab)
        self.tabs.currentChanged.connect(self.view.changeTab)

        self.split_button = QtWidgets.QToolButton()
        self.split_button.setAutoRaise(True)
        self.split_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.split_button.setIcon(QtGui.QIcon.fromTheme("view-split-left-right"))
        self.split_button.addAction(self.view.viewspace.action_split_horz)
        self.split_button.addAction(self.view.viewspace.action_split_vert)
        self.split_button.addAction(self.view.viewspace.action_close_view)
        self.split_button.installEventFilter(self.view)

        # Layout the windgets
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        layout.addWidget(line)
        layout.addWidget(self.tabs, 0, QtCore.Qt.AlignLeft)
        layout.addStretch(1)
        layout.addWidget(self.split_button)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        layout.addWidget(line)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    mw = QtWidgets.QMainWindow()
    viewspace = ViewSpace()
    w = QtWidgets.QWidget()
    w.setMinimumSize(800, 600)
    lo = QtWidgets.QVBoxLayout()
    lo.addWidget(viewspace)
    w.setLayout(lo)
    mw.setCentralWidget(w)
    view = View(viewspace)
    viewspace.addWidget(view)
    for i in range(0, 5):
        view.addTab(QtWidgets.QGroupBox(str(i)), str(i))
    mw.show()
    app.exec_()
