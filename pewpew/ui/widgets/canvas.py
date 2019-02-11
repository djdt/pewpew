from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from pewpew.lib.formatter import formatIsotope
from pewpew.lib.plotimage import plotLaserImage

from pewpew.lib.laser import LaserData
from matplotlib.backend_bases import MouseEvent, LocationEvent

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter

from typing import Callable, Dict, List, Tuple
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


class DragSelector(QtWidgets.QRubberBand):
    def __init__(
        self,
        button: int = 1,
        callback: Callable = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(QtWidgets.QRubberBand.Rectangle, parent)
        self.button = button
        self.callback = callback
        self.extent = (0, 0, 0, 0)
        self.origin = QtCore.QPoint()
        self.cids: List[int] = []

    def _press(self, event: MouseEvent) -> None:
        self.event_press = event
        if event.button != self.button:
            return
        self.origin = event.guiEvent.pos()
        self.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
        self.show()

    def _move(self, event: MouseEvent) -> None:
        if event.button != self.button:
            return
        self.setGeometry(QtCore.QRect(self.origin, event.guiEvent.pos()).normalized())

    def _release(self, event: MouseEvent) -> None:
        self.event_release = event
        if event.button != self.button:
            return

        trans = self.axes.transData.inverted()
        x1, y1 = trans.transform_point((self.event_press.x, self.event_press.y))
        x2, y2 = trans.transform_point((self.event_release.x, self.event_release.y))

        # Order points
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Bound in axes limits
        lx1, lx2 = self.axes.get_xlim()
        ly1, ly2 = self.axes.get_ylim()
        x1 = max(lx1, min(lx2, x1))
        x2 = max(lx1, min(lx2, x2))
        y1 = max(ly1, min(ly2, y1))
        y2 = max(ly1, min(ly2, y2))

        self.extent = x1, x2, y1, y2

        if self.callback is not None:
            self.callback(self.event_press, self.event_release)
        self.hide()

    def activate(self, axes: Axes, callback: Callable = None) -> None:
        self.axes = axes
        self.callback = callback
        self.cids = [
            self.parent().mpl_connect("button_press_event", self._press),
            self.parent().mpl_connect("motion_notify_event", self._move),
            self.parent().mpl_connect("button_release_event", self._release),
        ]

    def deactivate(self) -> None:
        for cid in self.cids:
            self.parent().mpl_disconnect(cid)

    def close(self) -> None:
        self.deactivate()
        super().close()


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

    def plot(self, laser: LaserData, isotope: str, viewconfig: dict) -> None:
        # Get the trimmed and calibrated data
        data = laser.get(isotope, calibrated=True)
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

        self.extent = laser.extent()

        # Plot the image
        self.image = plotLaserImage(
            self.figure,
            self.ax,
            data,
            aspect=laser.aspect(),
            cmap=viewconfig["cmap"]["type"],
            colorbar=self.options["colorbar"],
            colorbarpos="bottom",
            colorbartext=str(laser.calibration[isotope]["unit"]),
            extent=self.extent,
            fontsize=viewconfig["font"]["size"],
            interpolation=viewconfig["interpolation"].lower(),
            label=self.options["label"],
            labeltext=formatIsotope(isotope, fstring="$^{{{mass}}}${element}"),
            scalebar=self.options["scalebar"],
            vmax=viewconfig["cmap"]["range"][1],
            vmin=viewconfig["cmap"]["range"][0],
            xaxis=True,
            xaxisticksize=laser.config["speed"],
        )
        if self.view == (0.0, 0.0, 0.0, 0.0):
            self.view = self.extent
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

    def getView(self, inverted: bool = False) -> Tuple[float, float, float, float]:
        x1, x2, y1, y2 = self.view
        if inverted:
            y2, y1 = self.extent[3] - y1, self.extent[3] - y2
        return (x1, x2, y1, y2)

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
            draw = False
            if x1 > xmin and x2 < xmax:
                self.ax.set_xlim(x1, x2)
                draw = True
            if y1 > ymin and y2 < ymax:
                self.ax.set_ylim(y1, y2)
                draw = True
            if draw:
                self.draw()

    def updateStatusBar(self, e: MouseEvent) -> None:
        if e.inaxes == self.ax:
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
