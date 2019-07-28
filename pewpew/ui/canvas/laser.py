from PySide2 import QtGui, QtWidgets
import numpy as np
import copy

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross, KrissKrossConfig

from pewpew.ui.canvas.basic import BasicCanvas
from pewpew.ui.canvas.interactive import InteractiveCanvas
from pewpew.ui.canvas.widgets import (
    LassoImageSelectionWidget,
    RectangleImageSelectionWidget,
)

from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import Tuple
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseEvent, LocationEvent
from matplotlib.patheffects import Normal, SimpleLineShadow
from matplotlib.widgets import _SelectorWidget, RectangleSelector


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
        # if self.viewconfig["filtering"]["type"] != "None":
        #     filter_type, window, threshold = (
        #         self.viewconfig["filtering"][x] for x in ["type", "window", "threshold"]
        #     )
        #     if filter_type == "Rolling mean":
        #         # rolling_mean_filter(data, window, threshold)
        #         data = rolling_mean_filter(data, window, threshold)
        #     elif filter_type == "Rolling median":
        #         data = rolling_median_filter(data, window, threshold)

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
            if layer is None or not isinstance(laser.config, KrissKrossConfig)
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
        self.state = set(["move"])
        self.button = 1

        shadow = self.palette().color(QtGui.QPalette.Shadow)
        highlight = self.palette().color(QtGui.QPalette.Highlight)
        lineshadow = SimpleLineShadow(
            offset=(0.5, -0.5), alpha=0.66, shadow_color=shadow.name()
        )
        self.rectprops = {
            "edgecolor": highlight.name(),
            "facecolor": None,
            "alpha": 0.33,
        }
        self.lineprops = {
            "color": highlight.name(),
            "linestyle": "--",
            "path_effects": [lineshadow, Normal()],
        }
        self.mask_rgba = np.array(
            [highlight.red(), highlight.green(), highlight.blue(), 255 * 0.5],
            dtype=np.uint8,
        )

        self.selector: _SelectorWidget = None

    def redrawFigure(self) -> None:
        super().redrawFigure()
        if hasattr(self, "rectangle_selector"):
            self.rectangle_selector.ax = self.ax
        if hasattr(self, "lasso_selector"):
            self.lasso_selector.ax = self.ax

    def drawLaser(self, laser: Laser, name: str, layer: int = None) -> None:
        super().drawLaser(laser, name, layer)
        # Save some variables for the status bar
        self.px, self.py = (
            (laser.config.pixel_width(), laser.config.pixel_height())
            if layer is None or not isinstance(laser.config, KrissKrossConfig)
            else (laser.config.layer_pixel_width(), laser.config.layer_pixel_height())
        )
        self.ps = laser.config.speed

    def startLassoSelection(self) -> None:
        self.clearSelection()
        self.state.add("selection")
        self.selector = LassoImageSelectionWidget(
            self.image,
            self.mask_rgba,
            useblit=True,
            button=self.button,
            lineprops=self.lineprops,
        )
        self.selector.set_active(True)

    def startRectangleSelection(self) -> None:
        self.clearSelection()
        self.state.add("selection")
        self.selector = RectangleImageSelectionWidget(
            self.image,
            self.mask_rgba,
            useblit=True,
            button=self.button,
            rectprops=self.rectprops,
        )
        self.selector.set_active(True)

    def clearSelection(self) -> None:
        if self.selector is not None:
            self.selector.set_active(False)
            self.selector.set_visible(False)
            self.selector.update()
        self.selector = None

    def ignore_event(self, event: LocationEvent) -> bool:
        if event.name in ["scroll_event", "key_press_event"]:
            return True
        elif (
            event.name in ["button_press_event", "button_release_event"]
            and event.button != self.button
        ):
            return True

        if event.inaxes != self.ax:
            return True

        return super().ignore_event(event)

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
        self.selector = RectangleSelector(
            self.ax,
            self.zoom,
            useblit=True,
            drawtype="box",
            button=self.button,
            rectprops=self.rectprops,
        )
        self.selector.set_active(True)

    def zoom(self, press: MouseEvent, release: MouseEvent) -> None:
        self.clearSelection()
        self.state.discard("selection")
        self.view_limits = (press.xdata, release.xdata, press.ydata, release.ydata)
        self.state.add("zoom")

    def unzoom(self) -> None:
        self.state.discard("zoom")
        self.view_limits = self.extent
