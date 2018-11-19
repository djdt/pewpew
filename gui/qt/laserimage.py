from PyQt5 import QtCore, QtWidgets

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from util.laserimage import LaserImage
from util.plothelpers import coords2value

import os.path

class LaserImageDock(QtWidgets.QDockWidget):
    def __init__(self, laserdata, parent=None):

        self.laserdata = laserdata
        name = os.path.splitext(os.path.basename(self.laserdata.source))[0]

        super().__init__(f"{name}:{self.laserdata.isotope}", parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                         QtWidgets.QDockWidget.DockWidgetMovable)

        self.fig = Figure(frameon=False)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)

        self.setWidget(self.canvas)

        self.canvas.mpl_connect('motion_notify_event', self.updateStatusBar)
        self.canvas.mpl_connect('axes_leave_event', self.clearStatusBar)

    def draw(self, laserdata=None):
        self.ax.clear()

        if laserdata is not None:
            self.laserdata = laserdata

        self.lase = LaserImage(self.fig, self.ax, self.laserdata.calibrated(),
                               label=self.laserdata.isotope,
                               aspect=self.laserdata.aspect(),
                               extent=self.laserdata.extent())
        self.fig.tight_layout()
        self.canvas.draw()

    def updateStatusBar(self, e):
        # TODO make sure no in the color bar axes
        if e.inaxes == self.ax:
            x, y = e.xdata, e.ydata
            v = coords2value(self.lase.im, x, y)
            self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v}]")

    def clearStatusBar(self, e):
        self.window().statusBar().clearMessage()

    def close(self):
        super().close()
        self.destroy()
