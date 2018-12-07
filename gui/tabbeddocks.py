from PyQt5 import QtCore, QtWidgets


class TabbedDocks(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setDockNestingEnabled(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    def addDockWidget(self, dock, area=QtCore.Qt.LeftDockWidgetArea):
        super().addDockWidget(area, dock)
        # Find a child dock in same area to tab
        docks = self.findChildren(QtWidgets.QDockWidget)
        for d in docks:
            if self.dockWidgetArea(d) == area and d != dock:
                self.tabifyDockWidget(d, dock)
                return

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
