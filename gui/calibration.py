from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.lines import Line2D

from gui.canvas import Canvas
from util.laserimage import plotLaserImage


class CalibrationWidget(QtWidgets.QDialog):
    def __init__(self, laser, parent=None):
        super().__init__(parent)

        self.laser = laser
        self.canvas = Canvas(self)

        layout_form = QtWidgets.QFormLayout()
        layout_horz = QtWidgets.QHBoxLayout()
        layout_main = QtWidgets.QVBoxLayout()
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Close
        )

    def draw(self):

        isotope = self.combo_isotope.currentText()
        viewconfig = self.window().viewconfig

        self.image = plotLaserImage(
            self.canvas.fig,
            self.canvas.ax,
            self.laser.get(isotope, calibrated=True, trimmed=True),
            scalebar=False,
            cmap=viewconfig["cmap"],
            interpolation=viewconfig["interpolation"],
            vmin=viewconfig["cmap_range"][0],
            vmax=viewconfig["cmap_range"][1],
            aspect=self.laser.aspect(),
            extent=self.laser.extent(trimmed=True),
        )

        from matplotlib.lines import Line2D

        for i in range(0, 1.0, 0.1):
            rect = Line2D(
                (0.0, 1.0),
                (i, i),
                transform=self.ax.transAxes,
                linewidth=5.0,
                color="white",
            )
            text = 
            self.canvas.ax.add_artist(rect)

        self.canvas.draw()
