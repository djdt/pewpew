from PyQt5 import QtCore, QtWidgets
import numpy as np

from pewpew.lib.laser import Laser

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class StatsDialog(QtWidgets.QDialog):
    def __init__(self, laser: Laser, viewconfig: dict, parent: QtWidgets.QWidget = None):
        self.laser = laser
        self.viewconfig = viewconfig
        super().__init__(parent)

        fig = Figure(frameon=False, tight_layout=True, figsize=(5, 2), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(fig)

        self.canvas.setParent(parent)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.addItems(self.laser.isotopes())
        self.combo_isotope.currentIndexChanged.connect(self.onComboIsotope)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.close)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.combo_isotope, 1, QtCore.Qt.AlignRight)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.plot()

    def onComboIsotope(self) -> None:
        self.plot()

    def plot(self) -> None:
        data = self.laser.data[self.combo_isotope.currentText()].data
        # vmin, vmax = self.viewconfig['cmap']['range']
        # if isinstance(vmin, str):
        #     vmin = np.percentile(data, float(vmin.rstrip("%")))
        # if isinstance(vmax, str):
        #     vmax = np.percentile(data, float(vmax.rstrip("%")))
        self.ax.hist(data.ravel(), bins='auto', color='black')
        self.canvas.draw()
