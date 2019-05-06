from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from pewpew.lib.plotimage import plot_laser_data

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross
from matplotlib.backend_bases import MouseEvent, LocationEvent

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter

from pewpew.ui.widgets.dragselector import DragSelector

from typing import Dict, List
from matplotlib.image import AxesImage


# TODO emit signal to update status bar in actual widgets
class Canvas(FigureCanvasQTAgg):
    def __init__(
        self, connect_mouse_events: bool = True, parent: QtWidgets.QWidget = None
    ) -> None:
        fig = Figure(frameon=False, tight_layout=True, figsize=(5, 5), dpi=100)
        super().__init__(fig)
        self.ax = self.figure.add_subplot(111)
        self.image = AxesImage(self.ax)

        self.extent = (0.0, 0.0, 0.0, 0.0)
        self.view = (0.0, 0.0, 0.0, 0.0)

        self.options = {"colorbar": True, "scalebar": True, "label": True}
        self.selector = DragSelector(parent=self)

        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.events: Dict[str, List[int]] = {}
        if connect_mouse_events:
            self.connectEvents("status")

    def close(self) -> None:
        self.clearStatusBar()
        super().close()

    def plot(self, laser: Laser, name: str, viewconfig: dict) -> None:
        # Get the trimmed and calibrated data
        kwargs = {"calibrate": viewconfig['calibrate']}
        if isinstance(laser, KrissKross):
            kwargs['flat'] = True
        data = laser.get(name, **kwargs)
        unit = str(laser.data[name].calibration.unit) if viewconfig["calibrate"] else ""
        # Filter if required
        if viewconfig["filtering"]["type"] != "None":
            filter_type, window, threshold = (
                viewconfig["filtering"][x] for x in ["type", "window", "threshold"]
            )
            if filter_type == "Rolling mean":
                # rolling_mean_filter(data, window, threshold)
                data = rolling_mean_filter(data, window, threshold)
            elif filter_type == "Rolling median":
                data = rolling_median_filter(data, window, threshold)

        # If the laser extent has changed (i.e. config edited) then reset the view
        x = data.shape[1] * laser.config.pixel_width()
        y = data.shape[0] * laser.config.pixel_height()
        extent = (0.0, x, 0.0, y)
        if self.extent != extent:
            self.extent = extent
            self.view = (0.0, 0.0, 0.0, 0.0)

        # Plot the image
        self.image = plot_laser_data(
            self.figure,
            self.ax,
            data,
            aspect=laser.config.aspect(),
            cmap=viewconfig["cmap"]["type"],
            colorbar="bottom" if self.options["colorbar"] else None,
            colorbar_label=unit,
            colorbar_range=viewconfig["cmap"]["range"],
            extent=self.extent,
            fontsize=viewconfig["font"]["size"],
            interpolation=viewconfig["interpolation"].lower(),
            alpha=viewconfig["alpha"],
            label=name if self.options["label"] else None,
            scalebar="upper right" if self.options["scalebar"] else None,
        )
        if self.view == (0.0, 0.0, 0.0, 0.0):
            self.view = self.extent
            self.draw()
        else:
            self.setView(*self.view)

    def clear(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def setView(self, x1: float, x2: float, y1: float, y2: float) -> None:
        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y1, y2)
        self.view = (x1, x2, y1, y2)
        self.draw()

    def unzoom(self) -> None:
        self.setView(*self.extent)
        self.disconnectEvents("drag")

    def zoom(self, press: MouseEvent, release: MouseEvent) -> None:
        xmin, xmax, ymin, ymax = self.selector.extent
        if xmin == xmax or ymin == ymax:  # Invalid
            return
        self.setView(xmin, xmax, ymin, ymax)
        self.selector.deactivate()
        self.connectEvents("drag")

    def connectEvents(self, key: str) -> None:
        if key == "status":
            self.events["status"] = [
                self.mpl_connect("motion_notify_event", self.updateStatusBar),
                self.mpl_connect("axes_leave_event", self.clearStatusBar),
            ]
        elif key == "drag":
            self.events["drag"] = [
                self.mpl_connect("button_press_event", self.dragStart),
                self.mpl_connect("motion_notify_event", self.dragMove),
            ]

    def disconnectEvents(self, key: str) -> None:
        if key in self.events.keys():
            for cid in self.events[key]:
                self.mpl_disconnect(cid)

    def startZoom(self) -> None:
        self.selector.activate(self.ax, self.zoom)
        self.disconnectEvents("drag")

    def dragStart(self, event: MouseEvent) -> None:
        if event.inaxes == self.ax and event.button == 1:
            self.drag_origin = event.xdata, event.ydata

    def dragMove(self, event: MouseEvent) -> None:
        if event.inaxes == self.ax and event.button == 1:
            x1, x2 = self.ax.get_xlim()
            y1, y2 = self.ax.get_ylim()

            # Move in opposite direction to drag
            x1 += self.drag_origin[0] - event.xdata
            x2 += self.drag_origin[0] - event.xdata
            y1 += self.drag_origin[1] - event.ydata
            y2 += self.drag_origin[1] - event.ydata
            # Bound to image exents
            xmin, xmax, ymin, ymax = self.image.get_extent()
            view = self.view
            if x1 > xmin and x2 < xmax:
                view = (x1, x2, view[2], view[3])
            if y1 > ymin and y2 < ymax:
                view = (view[0], view[1], y1, y2)
            if view != self.view:
                self.setView(*view)

    def updateStatusBar(self, e: MouseEvent) -> None:
        if e.inaxes == self.ax and self.image._A is not None:
            x, y = e.xdata, e.ydata
            v = self.image.get_cursor_data(e)
            if self.window() is not None and self.window().statusBar() is not None:
                self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v:.2f}]")

    def clearStatusBar(self, e: LocationEvent = None) -> None:
        if self.window() is not None:
            self.window().statusBar().clearMessage()

    def sizeHint(self) -> QtCore.QSize:
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 250)
