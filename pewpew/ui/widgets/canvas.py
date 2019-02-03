from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from pewpew.lib.formatter import formatIsotope
from pewpew.lib.plotimage import plotLaserImage

from pewpew.lib.laser import LaserData
from matplotlib.backend_bases import MouseEvent, LocationEvent

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter


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

            self.mpl_connect("motion_notify_event", self.zoomDragMotion)
            self.mpl_connect("button_press_event", self.zoomDragBegin)
            self.mpl_connect("button_release_event", self.zoomDragEnd)

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

    def onMouseRelease(self, e: MouseEvent) -> None:
        if e.inaxes == self.ax:
            self.zoom_rect[1] = (e.xdata, e.ydata)
            # Reset to full size
            if self.zoom_rect[0] == self.zoom_rect[1]:
                self.zoomed = False
                extent = self.image.get_extent()
                self.ax.set_xlim(extent[0], extent[1])
                self.ax.set_ylim(extent[2], extent[3])
            # Zoom to rect
            else:
                self.zoomed = True
                self.ax.set_xlim(self.zoom_rect[0][0], self.zoom_rect[1][0])
                self.ax.set_ylim(self.zoom_rect[0][1], self.zoom_rect[1][1])
            self.draw()

    def zoomDragBegin(self, e: MouseEvent) -> None:
        if e.inaxes and e.button == 1:  # left mouse
            self.zoom_start = (e.xdata, e.ydata)
            self.rubberband_origin = e.guiEvent.pos()
            self.rubberband.setGeometry(
                QtCore.QRect(self.rubberband_origin, QtCore.QSize())
            )
            self.rubberband.show()

    def zoomDragMotion(self, e: MouseEvent) -> None:
        if e.inaxes and e.button == 1:
            if self.rubberband.isVisible():
                self.rubberband.setGeometry(
                    QtCore.QRect(self.rubberband_origin, e.guiEvent.pos()).normalized()
                )

    def zoomDragEnd(self, e: MouseEvent) -> None:
        if e.inaxes:  # left mouse
            self.zoom_end = (e.xdata, e.ydata)

            if self.zoom_start == self.zoom_end:  # No movement
                self.zoomed = False
                extent = self.image.get_extent()
                self.ax.set_xlim(extent[0], extent[1])
                self.ax.set_ylim(extent[2], extent[3])
            else:
                self.zoomed = True
                if self.zoom_start[0] < self.zoom_end[0]:
                    self.ax.set_xlim(self.zoom_start[0], self.zoom_end[0])
                else:
                    self.ax.set_xlim(self.zoom_end[0], self.zoom_start[0])
                if self.zoom_start[1] < self.zoom_end[1]:
                    self.ax.set_ylim(self.zoom_start[1], self.zoom_end[1])
                else:
                    self.ax.set_ylim(self.zoom_end[1], self.zoom_start[1])

            self.draw()

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
