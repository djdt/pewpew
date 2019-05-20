import copy
from PyQt5 import QtWidgets
import numpy as np

from laserlib.laser import Laser
from laserlib.krisskross import KrissKross

from pewpew.lib.calc import rolling_mean_filter, rolling_median_filter

from pewpew.ui.canvas.basic import BasicCanvas

from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import Tuple
from matplotlib.image import AxesImage


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
        self.view_limits = (0.0, 0.0, 0.0, 0.0)

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
        self.cax.tick_params(labelsize=self.viewconfig["font"]["size"])

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

    def updateView(self) -> None:
        x1, x2, y1, y2 = self.view_limits
        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y1, y2)
        self.draw_idle()
