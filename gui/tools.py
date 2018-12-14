from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from matplotlib.lines import Line2D

from gui.canvas import Canvas
from gui.imagedock import LaserImageDock

from util.laserimage import plotLaserImage


class CalibrationTool(QtWidgets.QDialog):
    def __init__(self, laser, parent=None):
        super().__init__(parent)

        self.laser = laser
        self.canvas = Canvas(parent=self)

        self.button_laser = QtWidgets.QPushButton("Select &Image...")
        self.button_laser.pressed.connect(self.onButtonLaser)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.onComboIsotope)

        label_isotope = QtWidgets.QLabel("Isotope:")

        layout_bar = QtWidgets.QHBoxLayout()
        layout_bar.addWidget(label_isotope, 1, QtCore.Qt.AlignRight)
        layout_bar.addWidget(self.combo_isotope)

        self.lineedit_levels = QtWidgets.QLineEdit()
        self.lineedit_levels.setText("5")
        self.lineedit_levels.setValidator(
            QtGui.QIntValidator(0, 20, self.lineedit_levels)
        )

        self.table = QtWidgets.QTableWidget()

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Image:", self.button_laser)
        layout_form.addRow("Calibration Levels:", self.lineedit_levels)
        layout_form.addRow(self.table)

        layout_canvas = QtWidgets.QVBoxLayout()
        layout_canvas.addWidget(self.canvas)
        layout_canvas.addLayout(layout_bar)

        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_form)
        layout_horz.addLayout(layout_canvas)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Close
        )

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_horz)
        layout_main.addWidget(buttons)
        self.setLayout(layout_main)

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
        for i in np.linspace(0, 1.0, 7):
            rect = Line2D(
                (0.0, 1.0),
                (i, i),
                transform=self.canvas.ax.transAxes,
                linewidth=2.0,
                color="white",
            )
            self.canvas.ax.add_artist(rect)
            self.canvas.ax.annotate(
                f"{i}",
                (0.0, i),
                xytext=(10, 0),
                textcoords="offset points",
                xycoords="axes fraction",
                horizontalalignment="left",
                verticalalignment="center",
                color="white",
            )

        self.canvas.draw()

    def onComboIsotope(self, text):
        pass

    def onButtonLaser(self):
        pass
