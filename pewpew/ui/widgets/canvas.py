from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from pewpew.lib.formatter import formatIsotope
from pewpew.lib.plotimage import plotLaserImage

from pewpew.lib.laser import LaserData
from matplotlib.backend_bases import MouseEvent, LocationEvent

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter

from typing import Tuple


class Canvas(FigureCanvasQTAgg):
    def __init__(
        self, connect_mouse_events: bool = True, parent: QtWidgets.QWidget = None
    ) -> None:
        fig = Figure(frameon=False, tight_layout=True, figsize=(5, 5), dpi=100)
        super().__init__(fig)
        self.ax = self.figure.add_subplot(111)
        self.image = np.array([], dtype=np.float64)

        self.use_colorbar = True
        self.use_scalebar = True
        self.use_label = True

        self.zoomed = False
        self.rubberband = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        if connect_mouse_events:
            self.mpl_connect("motion_notify_event", self.updateStatusBar)
            self.mpl_connect("axes_leave_event", self.clearStatusBar)

            self.startZoom()

    def close(self) -> None:
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
            self.figure,
            self.ax,
            data,
            aspect=laser.aspect(),
            cmap=viewconfig["cmap"]["type"],
            colorbar=self.use_colorbar,
            colorbarpos="bottom",
            colorbartext=str(laser.calibration[isotope]["unit"]),
            extent=laser.extent(trimmed=True),
            fontsize=viewconfig["font"]["size"],
            interpolation=viewconfig["interpolation"].lower(),
            label=self.use_label,
            labeltext=formatIsotope(isotope, fstring="$^{{{mass}}}${element}"),
            scalebar=self.use_scalebar,
            vmax=viewconfig["cmap"]["range"][1],
            vmin=viewconfig["cmap"]["range"][0],
            xaxis=True,
            xaxisticksize=laser.config["speed"],
        )

    def clear(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def zoom(
        self, start: Tuple[float, float] = None, end: Tuple[float, float] = None
    ) -> None:
        extent = self.image.get_extent()
        if start is None or end is None:  # No movement
            xlim = extent[0], extent[1]
            ylim = extent[2], extent[3]
        else:
            # Correct order
            xlim = min(start[0], end[0]), max(start[0], end[0])
            ylim = min(start[1], end[1]), max(start[1], end[1])
            # Bound
            xlim = max(extent[0], xlim[0]), min(extent[1], xlim[1])
            ylim = max(extent[2], ylim[0]), min(extent[3], ylim[1])

        self.ax.set_xlim(*xlim)
        self.ax.set_xlim(*ylim)

    def startZoom(self) -> None:
        self.zoom_events = [
            self.mpl_connect("button_press_event", self.mousePressZoom),
            self.mpl_connect("motion_notify_event", self.mouseDragZoom),
            self.mpl_connect("button_release_event", self.mouseReleaseZoom),
        ]

    def mousePressZoom(self, event: MouseEvent) -> None:
        if event.inaxes == self.ax and event.button == 1:  # left mouse
            self.zoom_start = (event.xdata, event.ydata)
            self.rubberband_origin = event.guiEvent.pos()
            self.rubberband.setGeometry(
                QtCore.QRect(self.rubberband_origin, QtCore.QSize())
            )
            self.rubberband.show()

    def mouseDragZoom(self, event: MouseEvent) -> None:
        if event.button == 1:
            if self.rubberband.isVisible():
                self.rubberband.setGeometry(
                    QtCore.QRect(
                        self.rubberband_origin, event.guiEvent.pos()
                    ).normalized()
                )

    def mouseReleaseZoom(self, event: MouseEvent) -> None:
        if event.inaxes == self.ax:
            zoom_end = (event.xdata, event.ydata)
            self.zoom(self.zoom_start, zoom_end)

            self.draw()

            for cid in self.zoom_events:
                self.mpl_disconnect(cid)

        self.rubberband.hide()

    def updateStatusBar(self, e: MouseEvent) -> None:
        if e.inaxes == self.ax:
            x, y = e.xdata, e.ydata
            v = self.image.get_cursor_data(e)
            if self.window() is not None and self.window().statusBar() is not None:
                self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v}]")

    def clearStatusBar(self, e: LocationEvent = None) -> None:
        if self.window() is not None:
            self.window().statusBar().clearMessage()

    def sizeHint(self) -> QtCore.QSize:
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 250)
