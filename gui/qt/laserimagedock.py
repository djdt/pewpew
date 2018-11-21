from PyQt5 import QtCore, QtWidgets

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from util.laserimage import LaserImage
from util.plothelpers import coords2value

import os.path


class ImageDock(QtWidgets.QDockWidget):
    def __init__(self, name, parent=None):

        super().__init__(name, parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                         QtWidgets.QDockWidget.DockWidgetMovable)

        self.fig = Figure(frameon=False)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.canvas.setMinimumSize(100, 100)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                  QtWidgets.QSizePolicy.MinimumExpanding)

        self.setWidget(self.canvas)

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw()


class LaserImageDock(ImageDock):
    def __init__(self, laserdata, parent=None):

        self.laserdata = laserdata
        name = os.path.splitext(os.path.basename(self.laserdata.source))[0]

        super().__init__(f"{name}:{self.laserdata.isotope}", parent)

        self.canvas.mpl_connect('motion_notify_event', self.updateStatusBar)
        self.canvas.mpl_connect('axes_leave_event', self.clearStatusBar)

    def draw(self, laserdata=None, cmap='magma'):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if laserdata is not None:
            self.laserdata = laserdata

        self.lase = LaserImage(self.fig, self.ax, self.laserdata.calibrated(),
                               colorbar='bottom', cmap=cmap,
                               label=self.laserdata.isotope,
                               aspect=self.laserdata.aspect(),
                               extent=self.laserdata.extent())
        super().draw()

    def updateStatusBar(self, e):
        # TODO make sure no in the color bar axes
        if e.inaxes == self.ax:
            x, y = e.xdata, e.ydata
            v = coords2value(self.lase.im, x, y)
            self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v}]")

    def clearStatusBar(self, e):
        self.window().statusBar().clearMessage()


class KrissKrossImageDock(QtWidgets.QDockWidget):
    def __init__(self, kkdata, parent=None):

        self.kkdata = kkdata
        name = os.path.splitext(os.path.basename(self.laserdata.source))[0]

        super().__init__(f"{name}:kk:{self.laserdata.isotope}", parent)

        self.canvas.mpl_connect('motion_notify_event', self.updateStatusBar)
        self.canvas.mpl_connect('axes_leave_event', self.clearStatusBar)

    def draw(self, kkdata=None, cmap='magma'):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        if kkdata is not None:
            self.kkdata = kkdata

        self.lase = LaserImage(self.fig, self.ax,
                               self.kkdata.calibrated(flat=True),
                               colorbar='bottom', cmap=cmap,
                               label=self.laserdata.isotope,
                               aspect=self.laserdata.aspect(),
                               extent=self.laserdata.extent())
        super().draw()

    def updateStatusBar(self, e):
        # TODO make sure no in the color bar axes
        if e.inaxes == self.ax:
            x, y = e.xdata, e.ydata
            v = coords2value(self.lase.im, x, y)
            self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v}]")

    def clearStatusBar(self, e):
        self.window().statusBar().clearMessage()
