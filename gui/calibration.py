from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class Canvas(FigureCanvasQTAgg):
    def __init__(self, fig, parent=None):
        super().__init__(fig)
        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

        self.rubber_band = None
        self.rubber_band_origin = QtCore.QSize()

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        return QtCore.QSize


class CalibrationWidget(QtWidgets.QDialog):
    def __init__(self, laser, parent=None):
        super().__init__(parent)

        self.fig = Figure(
            frameon=False, tight_layout=True, figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = Canvas(self.fig, self)

    def draw(self):
        pass
:
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        isotope = self.combo_isotope.currentText()
        viewconfig = self.window().viewconfig

        self.image = plotLaserImage(
            self.fig,
            self.ax,
            self.laser.get(isotope, calibrated=True, trimmed=True),
            colorbar='bottom',
            colorbarlabel=self.laser.calibration['units'].get(isotope, ""),
            label=isotopeFormat(isotope),
            fontsize=viewconfig['fontsize'],
            cmap=viewconfig['cmap'],
            interpolation=viewconfig['interpolation'],
            vmin=viewconfig['cmap_range'][0],
            vmax=viewconfig['cmap_range'][1],
            aspect=self.laser.aspect(),
            extent=self.laser.extent(trimmed=True))

        from matplotlib.lines import Line2D
        for i in range(0, 1.0, 0.1):
            rect = Line2D((0.0, 1.0), (i, i),
                          transform=self.ax.transAxes,
                          linewidth=5.0,
                          color='white')
            self.ax.add_artist(rect)

        self.canvas.draw()

