from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from util.laserimage import LaserImage
from util.plothelpers import coords2value

from gui.qt.configdialog import ConfigDialog
from gui.qt.savedialog import SaveDialog

import os.path


class ImageDock(QtWidgets.QDockWidget):
    def __init__(self, name, parent=None):

        super().__init__(name, parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                         QtWidgets.QDockWidget.DockWidgetMovable)

        # self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        self.fig = Figure(frameon=False, figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.canvas.setMinimumSize(100, 100)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                  QtWidgets.QSizePolicy.MinimumExpanding)

        self.setWidget(self.canvas)

        # Context menu actions
        self.action_save = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('document-save'), "Save", self)
        self.action_save.setStatusTip("Save selected data.")
        self.action_save.triggered.connect(self.onContextMenuSave)

        self.action_config = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('document-properties'), "Config", self)
        self.action_config.setStatusTip("Edit the LA-ICP-MS config.")
        self.action_config.triggered.connect(self.onContextMenuConfig)

        self.action_close = QtWidgets.QAction(
            QtGui.QIcon.fromTheme('edit-delete'), "Close", self)
        self.action_close.setStatusTip("Close the image.")
        self.action_close.triggered.connect(self.onContextMenuClose)

        self.canvas.mpl_connect('motion_notify_event', self.updateStatusBar)
        self.canvas.mpl_connect('axes_leave_event', self.clearStatusBar)

    def updateStatusBar(self, e):
        # TODO make sure no in the color bar axes
        if e.inaxes == self.ax:
            x, y = e.xdata, e.ydata
            v = coords2value(self.lase.im, x, y)
            self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v}]")

    def clearStatusBar(self, e):
        self.window().statusBar().clearMessage()

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw()

    def contextMenuEvent(self, event):
        context_menu = QtWidgets.QMenu(self)
        context_menu.addAction(self.action_save)
        context_menu.addAction(self.action_config)
        context_menu.addSeparator()
        context_menu.addAction(self.action_close)
        context_menu.exec(event.globalPos())

    # def onContextMenuOpen(self):
    #     pass

    def onContextMenuSave(self):
        pass

    def onContextMenuConfig(self):
        pass

    def onContextMenuClose(self):
        self.close()


class LaserImageDock(ImageDock):
    def __init__(self, laserdata, parent=None):

        self.laserdata = laserdata
        name = os.path.splitext(os.path.basename(self.laserdata.source))[0]
        super().__init__(f"{name}:{self.laserdata.isotope}", parent)

    def draw(self, laserdata=None, cmap='magma'):
        self.fig.clear()
        self.canvas.draw()
        self.ax = self.fig.add_subplot(111)

        if laserdata is not None:
            self.laserdata = laserdata

        self.lase = LaserImage(self.fig, self.ax, self.laserdata.calibrated(),
                               colorbar='bottom', cmap=cmap,
                               label=self.laserdata.isotope,
                               aspect=self.laserdata.aspect(),
                               extent=self.laserdata.extent())
        super().draw()

    def onContextMenuConfig(self):
        dlg = ConfigDialog(self.laserdata.config, parent=self)
        dlg.check_all.setEnabled(False)
        if dlg.exec():
            self.laserdata.config = dlg.form.config
            self.draw()

    def onContextMenuSave(self):
        dlg = SaveDialog(self)
        if dlg.exec():
            dlg.save(self.laserdata)

class KrissKrossImageDock(ImageDock):
    def __init__(self, kkdata, parent=None):

        self.kkdata = kkdata
        name = os.path.splitext(os.path.basename(self.kkdata.source))[0]
        super().__init__(f"{name}:kk:{self.kkdata.isotope}", parent)

    def draw(self, kkdata=None, cmap='magma'):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if kkdata is not None:
            self.kkdata = kkdata

        self.lase = LaserImage(self.fig, self.ax,
                               self.kkdata.calibrated(flat=True),
                               colorbar='bottom', cmap=cmap,
                               label=self.kkdata.isotope,
                               aspect=self.kkdata.aspect(),
                               extent=self.kkdata.extent())
        super().draw()

    def onContextMenuConfig(self):
        dlg = ConfigDialog(self.kkdata.config, parent=self)
        dlg.check_all.setEnabled(False)
        if dlg.exec():
            self.kkdata.config = dlg.form.config
            self.draw()

    def onContextMenuSave(self):
        pass
