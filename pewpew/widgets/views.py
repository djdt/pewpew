from PySide2 import QtCore, QtGui, QtWidgets

from typing import List


class ViewSpace(QtWidgets.QSplitter):
    numViewsChanged = QtCore.Signal()
    numTabsChanged = QtCore.Signal()

    def __init__(
        self,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(orientation, parent)
        self.active_view: "View" = None
        self.views: List[View] = []

        self.action_split_horz = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-split-left-right"), "Split &Vertical"
        )
        self.action_split_horz.triggered.connect(self.splitActiveHorizontal)
        self.action_split_vert = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-split-top-bottom"), "Split &Horizontal"
        )
        self.action_split_vert.triggered.connect(self.splitActiveVertical)
        self.action_close_view = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("view-close"), "Close View"
        )
        self.action_close_view.triggered.connect(self.closeActiveView)

        self.addWidget(self.createView())

    def createView(self) -> "View":
        view = View(self)
        view.numTabsChanged.connect(self.numTabsChanged)
        self.views.append(view)
        self.numViewsChanged.emit()
        self.setActiveView(view)
        return view

    def countViewTabs(self) -> int:
        widgets = 0
        for view in self.views:
            widgets += view.tabs.count()
        return widgets

    def closeActiveView(self) -> None:
        self.closeView(self.activeView())
        self.active_view = None

    def closeView(self, view: "View") -> None:
        if view is None:
            return

        if len(self.views) == 1:
            return

        splitter = view.parent()
        if splitter is None:
            return

        self.views.remove(view)
        self.numViewsChanged.emit()

        view.deleteLater()
        view.setParent(None)

        assert splitter.count() == 1

        if splitter != self:
            parent_splitter = splitter.parent()
            if parent_splitter is not None:
                index = parent_splitter.indexOf(splitter)
                sizes = parent_splitter.sizes()
                parent_splitter.insertWidget(index, splitter.widget(0))
                splitter.deleteLater()
                splitter.setParent(None)
                parent_splitter.setSizes(sizes)
        # Doesn't seem to enter here?
        elif isinstance(splitter.widget(0), QtWidgets.QSplitter):
            child_splitter = splitter.widget(0)
            sizes = child_splitter.sizes()
            splitter.setOrientation(child_splitter.orientation())
            splitter.addWidget(child_splitter.widget(0))
            splitter.addWidget(child_splitter.widget(0))
            child_splitter.deleteLater()
            child_splitter.setParent(None)
            splitter.setSizes(sizes)

    def splitActiveHorizontal(self) -> None:
        self.splitView()

    def splitActiveVertical(self) -> None:
        self.splitView(None, QtCore.Qt.Vertical)

    def splitView(
        self,
        view: "View" = None,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
    ) -> None:
        if view is None:
            view = self.activeView()

        splitter = view.parent()
        index = splitter.indexOf(view)

        if splitter.count() == 1:
            size = (splitter.sizes()[0] - splitter.handleWidth()) / 2.0
            splitter.setOrientation(orientation)
            new_view = self.createView()
            self.addWidget(new_view)
            splitter.insertWidget(index - 1, new_view)
            splitter.setSizes([size, size])
        else:
            new_splitter = QtWidgets.QSplitter(orientation)
            new_splitter.setChildrenCollapsible(False)

            sizes = splitter.sizes()

            new_splitter.addWidget(view)
            new_view = self.createView()
            new_splitter.addWidget(new_view)

            splitter.insertWidget(index, new_splitter)

            splitter.setSizes(sizes)
            new_size = (sum(new_splitter.sizes()) - new_splitter.handleWidth()) / 2.0
            new_splitter.setSizes([new_size, new_size])

    def activeView(self) -> "View":
        if self.active_view is None:
            view = self.views[0]
            self.active_view = view

        return self.active_view

    def setActiveView(self, view: "View") -> None:
        if self.active_view == view:
            return
        if self.active_view is not None:
            self.active_view.setActive(False)
        if view is not None:
            view.setActive(True)
        self.active_view = view

    def activeWidget(self) -> QtWidgets.QWidget:
        widget = self.activeView().activeWidget()
        if widget is None:
            for view in self.views:
                widget = view.activeWidget()
                if widget is not None:
                    break
        return widget

    def refresh(self) -> None:
        for view in self.views:
            view.refresh()


class View(QtWidgets.QWidget):
    numTabsChanged = QtCore.Signal()

    def __init__(self, viewspace: ViewSpace, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setAcceptDrops(True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.active = False
        self.viewspace = viewspace

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setFrameStyle(QtWidgets.QFrame.StyledPanel | QtWidgets.QFrame.Sunken)

        self.tabs = ViewTabBar(self)
        self.tabs.setDrawBase(False)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        self.tabs.tabMoved.connect(self.moveStackWidget)
        self.tabs.tabClosed.connect(self.removeTab)
        self.tabs.installEventFilter(self)

        self.titlebar = ViewTitleBar(self.tabs, self)

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.titlebar, 0)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def widgets(self) -> List[QtWidgets.QWidget]:
        return [self.stack.widget(i) for i in range(self.stack.count())]

    def activeWidget(self) -> QtWidgets.QWidget:
        if self.stack.count() == 0:
            return None
        return self.stack.widget(self.stack.currentIndex())

    def addTab(self, text: str, widget: QtWidgets.QWidget) -> int:
        index = self.tabs.addTab(text)
        self.stack.insertWidget(index, widget)
        self.numTabsChanged.emit()
        return index

    def insertTab(self, index: int, text: str, widget: QtWidgets.QWidget) -> int:
        index = self.tabs.insertTab(index, text)
        self.stack.insertWidget(index, widget)
        self.numTabsChanged.emit()
        return index

    def removeTab(self, index: int) -> None:
        self.tabs.removeTab(index)
        self.stack.removeWidget(self.stack.widget(index))
        self.numTabsChanged.emit()

    def moveStackWidget(self, ifrom: int, ito: int) -> None:
        self.stack.insertWidget(ito, self.stack.widget(ifrom))

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        self.viewspace.setActiveView(self)

    def setActive(self, active: bool) -> None:
        self.active = active
        if active:
            color = self.palette().color(QtGui.QPalette.Highlight).name()
        else:
            color = self.palette().color(QtGui.QPalette.Shadow).name()
        self.stack.setStyleSheet(
            f"QStackedWidget, QStackedWidget > QWidget {{ color: {color} }}"
        )

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj and event.type() == QtCore.QEvent.MouseButtonPress:
            self.viewspace.setActiveView(self)
        return False

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            self.tabs.dropEvent(event)

    def refresh(self) -> None:
        pass


class ViewTabBar(QtWidgets.QTabBar):
    tabClosed = QtCore.Signal(int)

    def __init__(self, view: View, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.view = view
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.setElideMode(QtCore.Qt.ElideRight)
        self.setExpanding(False)
        self.setTabsClosable(True)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        self.tabCloseRequested.connect(self.tabClosed)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> QtWidgets.QDialog:
        if event.button() != QtCore.Qt.LeftButton:
            return None
        index = self.tabAt(event.pos())
        if index == -1:
            return None
        dlg = QtWidgets.QInputDialog(self)
        dlg.setWindowTitle("Rename")
        dlg.setLabelText("Name:")
        dlg.setTextValue(self.tabText(index))
        dlg.setInputMode(QtWidgets.QInputDialog.TextInput)
        dlg.textValueSelected.connect(lambda s: self.setTabText(index, s))
        dlg.open()
        return dlg

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_start_pos = event.pos()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not event.buttons() & QtCore.Qt.LeftButton:
            return
        if (
            event.pos() - self.drag_start_pos
        ).manhattanLength() < QtWidgets.QApplication.startDragDistance():
            return
        index = self.tabAt(event.pos())
        if index == -1:
            return

        rect = self.tabRect(index)
        pixmap = QtGui.QPixmap(rect.size())
        self.render(pixmap, QtCore.QPoint(), QtGui.QRegion(rect))

        mime_data = QtCore.QMimeData()
        mime_data.setData(
            "application/x-pewpewtabbar", QtCore.QByteArray().number(index)
        )

        drag = QtGui.QDrag(self)
        drag.setMimeData(mime_data)
        drag.setPixmap(pixmap)
        drag.setDragCursor(
            QtGui.QCursor(QtCore.Qt.DragMoveCursor).pixmap(), QtCore.Qt.MoveAction
        )
        drag.exec_(QtCore.Qt.MoveAction)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-pewpewtabbar"):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        dest = self.tabAt(event.pos())
        src, ok = event.mimeData().data("application/x-pewpewtabbar").toInt()
        if ok and event.source() == self:
            self.moveTab(src, dest)
        elif ok and isinstance(event.source(), ViewTabBar):
            text = event.source().tabText(src)
            widget = event.source().view.stack.widget(src)

            event.source().view.removeTab(src)

            index = self.view.insertTab(dest, text, widget)
            self.setCurrentIndex(index)
        else:
            return

        event.acceptProposedAction()


class ViewTitleBar(QtWidgets.QWidget):
    def __init__(self, tabs: ViewTabBar, view: View):
        super().__init__(view)
        self.view = view

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
        layout.addWidget(tabs, 1)
        layout.addWidget(self.split_button)
        self.setLayout(layout)
