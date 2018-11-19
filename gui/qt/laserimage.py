from PyQt5 import QtCore, QtWidgets

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from util.laserimage import LaserImage
from util.plothelpers import coords2value


class LaserImageDock(QtWidgets.QDockWidget):
    def __init__(self, data, isotope, params, source, parent=None):

        self.data = data
        self.isotope = isotope
        self.params = params
        self.source = source

        super().__init__(isotope, parent)
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

    def draw(self, data=None):
        self.ax.clear()

        if data is not None:
            self.data = data

        img = (self.data - self.params.intercept) / self.params.gradient
        self.lase = LaserImage(self.fig, self.ax, img, label=self.isotope,
                               aspect=self.params.aspect(),
                               extent=self.params.extent(self.data.shape))
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
