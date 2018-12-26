from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from util.formatter import formatIsotope
from util.laserimage import plotLaserImage
from util.plothelpers import coords2value

from util.laser import LaserData
from matplotlib.backend_bases import MouseEvent, LocationEvent


class Canvas(FigureCanvasQTAgg):
    def __init__(
        self, connect_mouse_events: bool = True, parent: QtWidgets.QWidget = None
    ) -> None:
        self.fig = Figure(frameon=False, tight_layout=True, figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.image = np.array([], dtype=np.float64)

        self.use_colorbar = True
        self.use_scalebar = True
        self.use_label = True

        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        if connect_mouse_events:
            self.mpl_connect("motion_notify_event", self.updateStatusBar)
            self.mpl_connect("axes_leave_event", self.clearStatusBar)

    def close(self) -> None:
        self.mpl_disconncet("motion_notify_event")
        self.mpl_disconncet("axes_leave_event")
        self.clearStatusBar()
        super().close()

    def plot(self, laser: LaserData, isotope: str, viewconfig: dict) -> None:
        self.image = plotLaserImage(
            self.fig,
            self.ax,
            laser.get(isotope, calibrated=True, trimmed=True),
            colorbar=self.use_colorbar,
            colorbarpos="bottom",
            colorbartext=laser.calibration[isotope]["unit"],
            scalebar=self.use_scalebar,
            label=self.use_label,
            labeltext=formatIsotope(isotope, fstring="$^{{{mass}}}${element}"),
            fontsize=viewconfig["fontsize"],
            cmap=viewconfig["cmap"],
            interpolation=viewconfig["interpolation"],
            vmin=viewconfig["cmaprange"][0],
            vmax=viewconfig["cmaprange"][1],
            aspect=laser.aspect(),
            extent=laser.extent(trimmed=True),
        )

    def clear(self) -> None:
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

    def updateStatusBar(self, e: MouseEvent) -> None:
        if e.inaxes == self.ax:
            x, y = e.xdata, e.ydata
            v = coords2value(self.image, x, y)
            self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v}]")

    def clearStatusBar(self, e: LocationEvent = None) -> None:
        self.window().statusBar().clearMessage()

    def sizeHint(self) -> QtCore.QSize:
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 250)
