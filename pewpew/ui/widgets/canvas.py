from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from pewpew.lib.formatter import formatIsotope
from pewpew.lib.plotimage import plotLaserImage

from pewpew.lib.laser import LaserData
from matplotlib.backend_bases import MouseEvent, LocationEvent

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter

from typing import Callable, Dict, List
from matplotlib.axes import Axes


# TODO write custom selector


class DragSelector(QtWidgets.QRubberBand):
    def __init__(self, callback: Callable = None, parent: QtWidgets.QWidget = None):
        super().__init__(QtWidgets.QRubberBand.Rectangle, parent)
        self.callback = callback
        self.extent = (0, 0, 0, 0)
        self.origin = QtCore.QPoint()
        self.cids: List[int] = []

    def _press(self, event: MouseEvent) -> None:
        self.event_press = event
        self.origin = event.guiEvent.pos()
        self.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
        self.show()

    def _move(self, event: MouseEvent) -> None:
        self.setGeometry(QtCore.QRect(self.origin, event.guiEvent.pos()).normalized())

    def _release(self, event: MouseEvent) -> None:
        self.event_release = event

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
        self.image = np.array([], dtype=np.float64)

        self.options = {"colorbar": True, "scalebar": True, "label": True}
        self.dragger = DragSelector(parent=self)

        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        self.events: Dict[str, List[int]] = {}
        if connect_mouse_events:
            self.events["status"] = [
                self.mpl_connect("motion_notify_event", self.updateStatusBar),
                self.mpl_connect("axes_leave_event", self.clearStatusBar),
            ]

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
            colorbar=self.options["colorbar"],
            colorbarpos="bottom",
            colorbartext=str(laser.calibration[isotope]["unit"]),
            extent=laser.extent(trimmed=True),
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

    def clear(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def unzoom(self) -> None:
        xmin, xmax, ymin, ymax = self.image.get_extent()
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.draw()

    def zoom(self, press: MouseEvent, release: MouseEvent) -> None:
        if press.inaxes != self.ax and release.inaxes != self.ax:  # Outside
            return
        xmin, xmax, ymin, ymax = self.dragger.extent
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.dragger.deactivate()
        self.draw()

    def disconnectEvents(self, key: str) -> None:
        for cid in self.events[key]:
            self.mpl_disconnect(cid)
        self.events[key] = []

    def startZoom(self) -> None:
        # self.selector.onselect = self.zoom
        # self.selector = RectangleSelector(self.ax, self.zoom, drawtype="line")
        # self.selector.set_active(True)
        self.dragger.activate(self.ax, self.zoom)
        # self.events["zoom"] = [
        #     self.mpl_connect("button_press_event", self.startDragZoom),
        #     self.mpl_connect("motion_notify_event", self.dragZoom),
        #     self.mpl_connect("button_release_event", self.endDragZoom),
        # ]

    def startMovement(self) -> None:
        self.events["movement"] = [
            self.mpl_connect("button_press_event", self.startDragMovement),
            self.mpl_connect("motion_notify_event", self.dragMovement),
            self.mpl_connect("button_release_event", self.endDragMovement),
        ]

    def startDrag(self, event: MouseEvent) -> None:
        self.rubberband_origin = event.guiEvent.pos()
        self.rubberband.setGeometry(
            QtCore.QRect(self.rubberband_origin, QtCore.QSize())
        )
        self.rubberband.show()
        pass

    # def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
    #     if self.selector.is_active():
    #         self.rubberband_origin = event.pos()
    #         self.rubberband.setGeometry(
    #             QtCore.QRect(self.rubberband_origin, QtCore.QSize())
    #         )
    #     super().mousePressEvent(event)

    # def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
    #     if self.selector.is_active():
    #         self.rubberband.setGeometry(
    #             QtCore.QRect(self.rubberband_origin, event.guiEvent.pos()).normalized()
    #         )
    #     super().mouseMoveEvent(event)

    # def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
    #     if self.selector.is_active():
    #         self.rubberband.hide()
    #     super().mouseReleaseEvent(event)

    def stareDragZoom(self, event: MouseEvent) -> None:
        self.drag_start = None
        if event.inaxes == self.ax and event.button == 1:  # left mouse
            self.drag_start = (event.xdata, event.ydata)
            self.rubberband_origin = event.guiEvent.pos()
            self.rubberband.setGeometry(
                QtCore.QRect(self.rubberband_origin, QtCore.QSize())
            )
            self.rubberband.show()

    def dragZoom(self, event: MouseEvent) -> None:
        if event.button == 1:
            if self.rubberband.isVisible():
                self.rubberband.setGeometry(
                    QtCore.QRect(
                        self.rubberband_origin, event.guiEvent.pos()
                    ).normalized()
                )

    def endDragZoom(self, event: MouseEvent) -> None:
        if event.inaxes == self.ax and self.zoom_start is not None:
            drag_end = (event.xdata, event.ydata)
            self.zoom(self.drag_start, drag_end)

            self.disconnectEvents("zoom")

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
