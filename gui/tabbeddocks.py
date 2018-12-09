from PyQt5 import QtCore, QtWidgets


class TabbedDocks(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setDockNestingEnabled(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    def orderedDocks(self, docks):
        """Returns docks sorted by leftmost / topmost."""
        return sorted(
            docks,
            key=
            lambda x: (x.geometry().topLeft().x(), x.geometry().topLeft().y()))

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
                self.largestDock(
                    [d for d in docks if not d.visibleRegion().isEmpty()]),
                dock)
            dock.draw()
            dock.show()

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
