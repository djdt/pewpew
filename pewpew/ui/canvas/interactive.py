import copy
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross
from matplotlib.backend_bases import MouseEvent, LocationEvent

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter
from pewpew.lib.colormaps import maskAlphaMap

from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path
from matplotlib.patheffects import Normal, SimpleLineShadow

from typing import Dict, List
from matplotlib.image import AxesImage

from pewpew.ui.canvas.laser import LaserCanvas
from pewpew.ui import color


class InteractiveLaserCanvas(LaserCanvas):
    EVENTS = {
        "status": [
            ("motion_notify_event", "updateStatusBar"),
            ("axes_leave_event", "clearStatusBar"),
        ],
        "drag": [("motion_notify_event", "drag"), ("button_press_event", "startDrag")],
    }

    def __init__(
        self,
        viewconfig: dict,
        options: dict = None,
        connect_mouse_events: bool = True,
        parent: QtWidgets.QWidget = None,
    ) -> None:
        super().__init__(viewconfig, options, parent=parent)

        self.events: Dict[str, List[int]] = {}
        if connect_mouse_events:
            self.connectEvents("status")

        self.image_selection: AxesImage = None

        lineshadow = SimpleLineShadow(
            offset=(0.5, -0.5), alpha=0.66, shadow_color=color.dark
        )
        rectprops = {
            "edgecolor": color.dark,
            "facecolor": color.highlight,
            "alpha": 0.33,
        }
        lineprops = {
            "color": color.highlight,
            "linestyle": "--",
            "path_effects": [lineshadow, Normal()],
        }

        self.rectangle_selector = RectangleSelector(
            self.ax,
            None,
            button=1,
            useblit=True,
            minspanx=5,
            minspany=5,
            rectprops=rectprops,
        )
        self.rectangle_selector.set_active(False)
        self.lasso_selector = LassoSelector(
            self.ax, None, button=1, useblit=True, lineprops=lineprops
        )
        self.lasso_selector.set_active(False)

    def redrawFigure(self) -> None:
        super().redrawFigure()
        self.rectangle_selector.ax = self.ax
        self.lasso_selector.ax = self.ax

    def connectEvents(self, key: str) -> None:
        events = self.EVENTS[key]
        cids = [self.mpl_connect(type, getattr(self, func)) for type, func in events]
        self.events[key] = cids

    def disconnectEvents(self, key: str) -> None:
        if key in self.events:
            for cid in self.events[key]:
                self.mpl_disconnect(cid)

    def startLassoSelection(self) -> None:
        self.rectangle_selector.set_active(False)
        self.lasso_selector.onselect = self.lassoSelection
        self.lasso_selector.set_active(True)
        self.disconnectEvents("drag")

    def lassoSelection(self, vertices: List[np.ndarray]) -> None:
        self.lasso_selector.set_active(False)
        self.draw_idle()

        data = self.image.get_array()
        x0, x1, y0, y1 = self.image.get_extent()
        ny, nx = data.shape
        # Calculate half pixel widths
        px, py = (x1 - x0) / nx / 2.0, (y0 - y1) / ny / 2.0

        # Grid of coords for the center of pixels
        x, y = np.meshgrid(
            np.linspace(x0 + px, x1 + px, nx, endpoint=False),
            np.linspace(y1 + py, y0 + py, ny, endpoint=False),
        )
        pix = np.vstack((x.flatten(), y.flatten())).T

        path = Path(vertices)
        ind = path.contains_points(pix, radius=2)

        mask = np.zeros(data.shape, dtype=np.uint8)
        mask.flat[ind] = 1

        self.image_selection = self.ax.imshow(
            mask, cmap=maskAlphaMap, extent=(x0, x1, y0, y1), alpha=0.33
        )

        if self.view_limits != self.image.get_extent():
            self.connectEvents("drag")

    def startRectangleSelection(self) -> None:
        self.lasso_selector.set_active(False)
        self.rectangle_selector.onselect = self.rectangleSelection
        self.rectangle_selector.set_active(True)
        pass

    def rectangleSelection(self) -> None:
        pass

    def startZoom(self) -> None:
        self.lasso_selector.set_active(False)
        self.rectangle_selector.onselect = self.zoom
        self.rectangle_selector.set_active(True)
        self.disconnectEvents("drag")

    def zoom(self, press: MouseEvent, release: MouseEvent) -> None:
        self.rectangle_selector.set_active(False)
        self.view_limits = (press.xdata, release.xdata, press.ydata, release.ydata)
        self.updateView()
        self.connectEvents("drag")

    def unzoom(self) -> None:
        self.view_limits = self.image.get_extent()
        self.updateView()
        self.disconnectEvents("drag")

    def startDrag(self, event: MouseEvent) -> None:
        if event.inaxes == self.ax and event.button == 1:
            self.drag_origin = event.xdata, event.ydata

    def drag(self, event: MouseEvent) -> None:
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
            view = self.view_limits
            if x1 > xmin and x2 < xmax:
                view = (x1, x2, view[2], view[3])
            if y1 > ymin and y2 < ymax:
                view = (view[0], view[1], y1, y2)
            # Update if changed
            if view != self.view_limits:
                self.view_limits = view
                self.updateView()

    def updateStatusBar(self, event: MouseEvent) -> None:
        if event.inaxes == self.ax and self.image._A is not None:
            x, y = event.xdata, event.ydata
            v = self.image.get_cursor_data(event)
            if self.window() is not None and self.window().statusBar() is not None:
                self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v:.2f}]")

    def clearStatusBar(self, event: LocationEvent) -> None:
        if self.window() is not None and self.window().statusBar() is not None:
            self.window().statusBar().clearMessage()
