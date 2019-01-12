from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from util.formatter import formatIsotope
from util.plotimage import plotLaserImage
from util.plothelpers import coords2value

from util.laser import LaserData
from matplotlib.backend_bases import MouseEvent, LocationEvent

from util.calc import rolling_mean_filter, rolling_median_filter


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
        # Get the trimmed and calibrated data
        data = laser.get(isotope, calibrated=True, trimmed=True)
        # Filter if required
        if viewconfig["filtering"]["type"] != "None":
            filter_type, window, threshold = (
                viewconfig["filtering"][x] for x in ["type", "window", "threshold"]
            )
            data = data.copy()
            if filter_type == "Rolling mean":
                rolling_mean_filter(data, window, threshold)
            elif filter_type == "Rolling median":
                rolling_median_filter(data, window, threshold)

        # Plot the image
        self.image = plotLaserImage(
            self.fig,
            self.ax,
            data,
            aspect=laser.aspect(),
            cmap=viewconfig["cmap"]["type"],
            colorbar=self.use_colorbar,
            colorbarpos="bottom",
            colorbartext=laser.calibration[isotope]["unit"],
            extent=laser.extent(trimmed=True),
            fontsize=viewconfig["font"]["size"],
            interpolation=viewconfig["interpolation"],
            label=self.use_label,
            labeltext=formatIsotope(isotope, fstring="$^{{{mass}}}${element}"),
            scalebar=self.use_scalebar,
            vmax=viewconfig["cmap"]["range"][1],
            vmin=viewconfig["cmap"]["range"][0],
            xaxis=True,
            xaxisticksize=laser.config["speed"],
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
