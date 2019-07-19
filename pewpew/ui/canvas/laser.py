import copy
from PyQt5 import QtGui, QtWidgets
import numpy as np

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter

from pewpew.ui.canvas.basic import BasicCanvas
from pewpew.ui.canvas.interactive import InteractiveCanvas

from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import List, Tuple
from matplotlib.image import AxesImage

from matplotlib.backend_bases import MouseEvent, LocationEvent

from pewpew.lib.colormaps import maskAlphaMap

from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path
from matplotlib.patheffects import Normal, SimpleLineShadow


class LaserCanvas(BasicCanvas):
    DEFAULT_OPTIONS = {
        "colorbar": True,
        "scalebar": True,
        "label": True,
        "colorbarpos": "bottom",
    }

    def __init__(
        self, viewconfig: dict, options: dict = None, parent: QtWidgets.QWidget = None
    ) -> None:
        super().__init__(parent=parent)
        self.viewconfig = viewconfig
        self.options = copy.deepcopy(LaserCanvas.DEFAULT_OPTIONS)
        if options is not None:
            self.options.update(options)

        self.redrawFigure()
        self.image: AxesImage = None

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        if self.image is None:
            return (0, 0, 0, 0)
        return self.image.get_extent()

    @property
    def view_limits(self) -> Tuple[float, float, float, float]:
        x0, x1, = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        return x0, x1, y0, y1

    @view_limits.setter
    def view_limits(self, limits: Tuple[float, float, float, float]) -> None:
        x0, x1, y0, y1 = limits
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.draw_idle()

    def redrawFigure(self) -> None:
        self.figure.clear()
        self.ax = self.figure.subplots()
        self.ax.set_facecolor("black")
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.autoscale(False)
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
        self.cax.tick_params(labelsize=self.viewconfig["font"]["size"])

    def drawData(
        self, data: np.ndarray, extent: Tuple[float, float, float, float]
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
            aspect="equal",
            origin="upper",
        )

    def drawLaser(self, laser: Laser, name: str, layer: int = None) -> None:
        # Get the trimmed and calibrated data
        kwargs = {"calibrate": self.viewconfig["calibrate"], "layer": layer}
        if isinstance(laser, KrissKross):
            kwargs["flat"] = True

        data = laser.get(name, **kwargs)
        unit = (
            str(laser.data[name].calibration.unit)
            if self.viewconfig["calibrate"]
            else ""
        )

        # Get extent
        extent = (
            laser.config.data_extent(data)
            if layer is None
            else laser.config.layer_data_extent(data)
        )

        # Only change the view if new or the laser extent has changed (i.e. conf edit)
        if self.image is None or self.extent != extent:
            self.view_limits = extent

        # If data is empty create a dummy data
        if data is None or data.size == 0:
            data = np.array([[0]], dtype=np.float64)

        self.drawData(data, extent)
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


class InteractiveLaserCanvas(LaserCanvas, InteractiveCanvas):
    def __init__(
        self,
        viewconfig: dict,
        options: dict = None,
        connect_mouse_events: bool = True,
        parent: QtWidgets.QWidget = None,
    ) -> None:
        super().__init__(viewconfig=viewconfig, options=options, parent=parent)

        self.status_bar = parent.window().statusBar()
        self.image_mask: AxesImage = None
        self.state = set(["move"])
        self.button = 1

        shadow_color = self.palette().color(QtGui.QPalette.Shadow).name()
        highlight_color = self.palette().color(QtGui.QPalette.Highlight).name()
        lineshadow = SimpleLineShadow(
            offset=(0.5, -0.5), alpha=0.66, shadow_color=shadow_color
        )
        rectprops = {
            "edgecolor": shadow_color,
            "facecolor": highlight_color,
            "alpha": 0.33,
        }
        lineprops = {
            "color": highlight_color,
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
        if hasattr(self, "rectangle_selector"):
            self.rectangle_selector.ax = self.ax
        if hasattr(self, "lasso_selector"):
            self.lasso_selector.ax = self.ax

    def drawData(
        self, data: np.ndarray, extent: Tuple[float, float, float, float]
    ) -> None:
        super().drawData(data, extent)
        if self.image_mask is not None:
            self.ax.add_image(self.image_mask)

    def drawMask(
        self, mask: np.ndarray, extent: Tuple[float, float, float, float]
    ) -> None:
        self.image_mask = self.ax.imshow(
            mask, cmap=maskAlphaMap, extent=extent, alpha=0.5
        )

    def drawLaser(self, laser: Laser, name: str, layer: int = None) -> None:
        super().drawLaser(laser, name, layer)
        # Save some variables for the status bar
        self.px, self.py = (
            (laser.config.pixel_width(), laser.config.pixel_height())
            if layer is None
            else (laser.config.layer_pixel_width(), laser.config.layer_pixel_height())
        )
        self.ps = laser.config.speed

    def startLassoSelection(self) -> None:
        self.clearSelection()
        self.state.add("selection")
        self.lasso_selector.onselect = self.lassoSelection
        self.lasso_selector.set_active(True)

    def lassoSelection(self, vertices: List[np.ndarray]) -> None:
        self.lasso_selector.set_active(False)
        self.state.discard("selection")

        data = self.image.get_array()
        x0, x1, y0, y1 = self.extent
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

        mask = np.zeros(data.shape, dtype=bool)
        mask.flat[ind] = True
        self.drawMask(mask, (x0, x1, y0, y1))
        self.draw_idle()

    def startRectangleSelection(self) -> None:
        self.clearSelection()
        self.state.add("selection")
        self.rectangle_selector.onselect = self.rectangleSelection
        self.rectangle_selector.set_active(True)

    def rectangleSelection(self, press: MouseEvent, release: MouseEvent) -> None:
        self.rectangle_selector.set_active(False)
        self.state.discard("selection")

        data = self.image.get_array()
        x0, x1, y0, y1 = self.extent
        ny, nx = data.shape
        # Calculate half pixel widths
        px, py = (x1 - x0) / nx / 2.0, (y0 - y1) / ny / 2.0

        # Grid of coords for the center of pixels
        x, y = np.meshgrid(
            np.linspace(x0 + px, x1 + px, nx, endpoint=False),
            np.linspace(y1 + py, y0 + py, ny, endpoint=False),
        )
        pix = np.vstack((x.flatten(), y.flatten())).T

        vertices = [
            (press.xdata, press.ydata),
            (release.xdata, press.ydata),
            (release.xdata, release.ydata),
            (press.xdata, release.ydata),
        ]
        path = Path(vertices)
        ind = path.contains_points(pix, radius=2)

        mask = np.zeros(data.shape, dtype=bool)
        mask.flat[ind] = True

        self.drawMask(mask, (x0, x1, y0, y1))
        self.draw_idle()

    def clearSelection(self) -> None:
        self.lasso_selector.set_active(False)
        self.rectangle_selector.set_active(False)
        if self.image_mask in self.ax.get_images():
            self.image_mask.remove()
            self.draw_idle()
        self.image_mask = None

    def ignore_event(self, event: LocationEvent) -> None:
        if event.name in ["scroll_event", "key_press_event"]:
            return True
        elif (
            event.name in ["button_press_event", "button_release_event"]
            and event.button != self.button
        ):
            return True

        if event.inaxes != self.ax:
            return True
        super().ignore_event(event)

    def press(self, event: MouseEvent) -> None:
        pass

    def release(self, event: MouseEvent) -> None:
        pass

    def move(self, event: MouseEvent) -> None:
        if (
            all(state in self.state for state in ["move", "zoom"])
            and "selection" not in self.state
            and event.button == self.button
        ):
            x1, x2, y1, y2 = self.view_limits
            xmin, xmax, ymin, ymax = self.extent
            dx = self.eventpress.xdata - event.xdata
            dy = self.eventpress.ydata - event.ydata

            # Move in opposite direction to drag
            if x1 + dx > xmin and x2 + dx < xmax:
                x1 += dx
                x2 += dx
            if y1 + dy > ymin and y2 + dy < ymax:
                y1 += dy
                y2 += dy
            self.view_limits = x1, x2, y1, y2

        # Update the status bar
        x, y = event.xdata, event.ydata
        v = self.image.get_cursor_data(event)
        unit = self.viewconfig["status_unit"]
        if unit == "row":
            x, y = int(x / self.px), int(y / self.py)
        elif unit == "second":
            x = event.xdata / self.ps
            y = 0
        self.status_bar.showMessage(f"{x:.4g},{y:.4g} [{v:.4g}]")

    def axis_enter(self, event: LocationEvent) -> None:
        pass

    def axis_leave(self, event: LocationEvent) -> None:
        self.status_bar.clearMessage()

    def startZoom(self) -> None:
        self.state.add("selection")
        self.lasso_selector.set_active(False)
        self.rectangle_selector.onselect = self.zoom
        self.rectangle_selector.set_active(True)

    def zoom(self, press: MouseEvent, release: MouseEvent) -> None:
        self.state.add("zoom")
        self.state.discard("selection")
        self.rectangle_selector.set_active(False)
        self.view_limits = (press.xdata, release.xdata, press.ydata, release.ydata)

    def unzoom(self) -> None:
        self.state.discard("zoom")
        self.view_limits = self.extent
