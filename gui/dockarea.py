from PyQt5 import QtCore, QtWidgets

from gui.events import MousePressRedirectFilter


class DockArea(QtWidgets.QMainWindow):

    mouseSelectFinished = QtCore.pyqtSignal('QWidget*')

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setDockNestingEnabled(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.mouse_select = False
        self.mouse_filter = MousePressRedirectFilter(self)

    def addDockWidgets(self, docks, area=QtCore.Qt.LeftDockWidgetArea):
        # Add a new dock widget
        super().addDockWidget(area, docks[0])
        origin = self.largestDock(self.visibleDocks())
        if origin is not None:
            self.smartSplitDock(origin, docks[0])
        docks[0].draw()
        docks[0].show()
        for dock in docks[1:]:
            super().addDockWidget(area, dock)
            self.smartSplitDock(
                self.largestDock([d for d in docks if not d.visibleRegion().isEmpty()]),
                dock,
            )
            dock.draw()
            dock.show()

    def orderedDocks(self, docks):
        """Returns docks sorted by leftmost / topmost."""
        return sorted(
            docks,
            key=lambda x: (x.geometry().topLeft().x(), x.geometry().topLeft().y()),
        )

    def largestDock(self, docks):
        largest = 0
        dock = None
        for d in self.orderedDocks(docks):
            size = d.size()
            if size.width() > largest:
                largest = size.width()
                dock = d
            if size.height() > largest:
                largest = size.height()
                dock = d
        return dock

    def smartSplitDock(self, first, second):
        size = first.size()
        minsize = second.minimumSizeHint()
        if size.width() > size.height() and size.width() > 2 * minsize.width():
            self.splitDockWidget(first, second, QtCore.Qt.Horizontal)
        elif size.height() > 2 * minsize.height():
            self.splitDockWidget(first, second, QtCore.Qt.Vertical)
        elif first != second:
            # Split only if there is enough space
            self.tabifyDockWidget(first, second)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.mouse_select is True:
            widget = None
            for dock in self.findChildren(QtWidgets.QDockWidget):
                if dock.underMouse():
                    widget = dock
                    break
            self.mouseSelectFinished.emit(widget)
            self.endMouseSelect()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if self.mouse_select is True:
            if event.key() == QtCore.Qt.Key_Escape:
                self.mouseSelectFinished.emit(None)
                self.endMouseSelect()

    def startMouseSelect(self):
        self.mouse_select = True
        for dock in self.findChildren(QtWidgets.QDockWidget):
            if hasattr(dock, 'canvas'):
                dock.canvas.installEventFilter(self.mouse_filter)

    def endMouseSelect(self):
        self.mouse_select = False
        for dock in self.findChildren(QtWidgets.QDockWidget):
            if hasattr(dock, 'canvas'):
                dock.canvas.removeEventFilter(self.mouse_filter)

    def tabifyAll(self, area=QtCore.Qt.LeftDockWidgetArea):
        docks = self.findChildren(QtWidgets.QDockWidget)
        # self.removeDockWidget(docks[0])
        self.addDockWidget(docks[0], area)

        for d in docks[1:]:
            self.tabifyDockWidget(docks[0], d)
            d.layout().invalidate()

    def visibleDocks(self, dock_type=QtWidgets.QDockWidget):
        docks = self.findChildren(dock_type)
        return [d for d in docks if not d.visibleRegion().isEmpty()]
