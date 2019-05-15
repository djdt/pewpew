from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross
from matplotlib.backend_bases import MouseEvent, LocationEvent

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter

from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.path import Path
from matplotlib.patheffects import Normal, SimpleLineShadow
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import Dict, List, Tuple
from matplotlib.image import AxesImage


class BasicCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QtWidgets.QWidget = None):
        fig = Figure(frameon=False, constrained_layout=True, figsize=(5, 5), dpi=100)
        super().__init__(fig)

        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 250)


class Canvas(BasicCanvas):
    DEFAULT_OPTIONS = {
        "colorbar": True,
        "scalebar": True,
        "label": True,
        "colorbarpos": "bottom",
    }

    def __init__(
        self,
        viewconfig: dict,
        options: dict = None,
        connect_mouse_events: bool = True,
        parent: QtWidgets.QWidget = None,
    ) -> None:
        super().__init__(parent)
        self.viewconfig = viewconfig
        self.options = Canvas.DEFAULT_OPTIONS
        if options is not None:
            self.options.update(options)
        self.image: AxesImage = None
        self.redrawFigure()

        self.view_limits = (0.0, 0.0, 0.0, 0.0)

        dark = self.palette().color(QtGui.QPalette.Shadow).name()
        highlight = self.palette().color(QtGui.QPalette.Highlight).name()
        lineshadow = SimpleLineShadow(offset=(0.5, -0.5), alpha=0.66, shadow_color=dark)
        rectprops = {"edgecolor": dark, "facecolor": highlight, "alpha": 0.33}
        lineprops = {
            "color": highlight,
            "linestyle": "--",
            "path_effects": [lineshadow, Normal()],
        }

        self.rectangle_selector = RectangleSelector(
            self.ax,
            self.zoom,
            button=1,
            useblit=True,
            minspanx=5,
            minspany=5,
            rectprops=rectprops,
            lineprops=lineprops,
        )
        self.rectangle_selector.set_active(False)
        self.lasso_selector = LassoSelector(
            self.ax, self.lasso, button=1, useblit=True, lineprops=lineprops
        )
        self.lasso_selector.set_active(False)

        self.events: Dict[str, List[int]] = {}
        if connect_mouse_events:
            self.connectEvents("status")

    def close(self) -> None:
        self.clearStatusBar()
        super().close()

    def redrawFigure(self) -> None:
        self.figure.clear()
        self.ax = self.figure.subplots()
        self.ax.axis("scaled")
        self.ax.set_facecolor("black")
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        if self.options["colorbar"]:
            div = make_axes_locatable(self.ax)
            self.cax = div.append_axes(self.options["colorbarpos"], size=0.1, pad=0.05)

    def drawColorbar(self, label: str) -> None:
        self.cax.clear()
        if self.options["colorbarpos"] in ["right", "left"]:
            orientation = "vertical"
        else:
            orientation = "horizontal"
        self.figure.colorbar(
            self.image,
            label=label,
            ax=self.ax,
            cax=self.cax,
            orientation=orientation,
            ticks=MaxNLocator(nbins=6),
        )

    def drawData(
        self, data: np.ndarray, extent: Tuple[float, float, float, float], aspect: float
    ) -> None:
        self.ax.clear()

        # Filter if required
        if self.viewconfig["filtering"]["type"] != "None":
            filter_type, window, threshold = (
                self.viewconfig["filtering"][x] for x in ["type", "window", "threshold"]
            )
            if filter_type == "Rolling mean":
                # rolling_mean_filter(data, window, threshold)
                data = rolling_mean_filter(data, window, threshold)
            elif filter_type == "Rolling median":
                data = rolling_median_filter(data, window, threshold)

        # Calculate the range
        rmin, rmax = self.viewconfig["cmap"]["range"]
        if isinstance(rmin, str):
            vmin = np.percentile(data, float(rmin.rstrip("%")))
        else:
            vmin = float(rmin)
        if isinstance(rmax, str):
            vmax = np.percentile(data, float(rmax.rstrip("%")))
        else:
            vmax = float(rmax)

        # Plot the image
        self.image = self.ax.imshow(
            data,
            cmap=self.viewconfig["cmap"]["type"],
            interpolation=self.viewconfig["interpolation"].lower(),
            alpha=self.viewconfig["alpha"],
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect=aspect,
            origin="upper",
        )

    def drawLaser(self, laser: Laser, name: str) -> None:
        # Get the trimmed and calibrated data
        kwargs = {"calibrate": self.viewconfig["calibrate"]}
        if isinstance(laser, KrissKross):
            kwargs["flat"] = True
        data = laser.get(name, **kwargs)
        unit = (
            str(laser.data[name].calibration.unit)
            if self.viewconfig["calibrate"]
            else ""
        )

        # Only change the view if new or the laser extent has changed (i.e. conf edit)
        extent = laser.config.data_extent(data)
        if self.image is None or self.image.get_extent() != extent:
            self.view_limits = extent

        # If data is empty create a dummy data
        if data is None or data.size == 0:
            data = np.array([[0]], dtype=np.float64)

        self.drawData(data, extent, laser.config.aspect())
        if self.options["colorbar"]:
            self.drawColorbar(unit)

        if self.options["label"]:
            text = AnchoredText(
                name,
                "upper left",
                pad=0.2,
                borderpad=0.1,
                frameon=False,
                prop={"color": "white", "size": self.viewconfig["font"]["size"]},
            )
            self.ax.add_artist(text)

        if self.options["scalebar"]:
            scalebar = ScaleBar(
                1.0,
                "um",
                location="upper right",
                frameon=False,
                color="white",
                font_properties={"size": self.viewconfig["font"]["size"]},
            )
            self.ax.add_artist(scalebar)

        # Return to zoom if extent not changed
        if self.view_limits != extent:
            self.updateView()

        self.draw()

    def updateView(self) -> None:
        x1, x2, y1, y2 = self.view_limits
        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y1, y2)
        self.draw_idle()

    def startLasso(self) -> None:
        self.lasso_selector.set_active(True)
        self.rectangle_selector.set_active(False)
        self.disconnectEvents("drag")

    def lasso(self, vertices: List[np.ndarray]) -> None:
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

        selected = np.zeros((*data.shape, 4), dtype=np.uint8)
        selected[:, :, 3].flat[~ind] = 128
        self.mask = self.ax.imshow(selected, extent=(x0, x1, y0, y1))

        if self.view_limits != self.image.get_extent():
            self.connectEvents("drag")

    def startZoom(self) -> None:
        self.rectangle_selector.set_active(True)
        self.lasso_selector.set_active(False)
        self.disconnectEvents("drag")

    def zoom(self, press: MouseEvent, release: MouseEvent) -> None:
        self.view_limits = (press.xdata, release.xdata, press.ydata, release.ydata)
        self.updateView()
        self.rectangle_selector.set_active(False)
        self.connectEvents("drag")

    def unzoom(self) -> None:
        self.view_limits = self.image.get_extent()
        self.updateView()
        self.disconnectEvents("drag")

    def connectEvents(self, key: str) -> None:
        if key == "status":
            self.events["status"] = [
                self.mpl_connect("motion_notify_event", self.updateStatusBar),
                self.mpl_connect("axes_leave_event", self.clearStatusBar),
            ]
        elif key == "drag":
            self.events["drag"] = [
                self.mpl_connect("button_press_event", self.startDrag),
                self.mpl_connect("motion_notify_event", self.drag),
            ]

    def disconnectEvents(self, key: str) -> None:
        if key in self.events.keys():
            for cid in self.events[key]:
                self.mpl_disconnect(cid)

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

    def updateStatusBar(self, e: MouseEvent) -> None:
        if e.inaxes == self.ax and self.image._A is not None:
            x, y = e.xdata, e.ydata
            v = self.image.get_cursor_data(e)
            if self.window() is not None and self.window().statusBar() is not None:
                self.window().statusBar().showMessage(f"{x:.2f},{y:.2f} [{v:.2f}]")

    def clearStatusBar(self, e: LocationEvent = None) -> None:
        if self.window() is not None:
            self.window().statusBar().clearMessage()
