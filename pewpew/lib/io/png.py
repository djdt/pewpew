from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pewpew.lib.plotimage import plotLaserImage
from pewpew.lib.laser import Laser
from pewpew.lib.formatter import formatIsotope

from typing import Tuple


def save(
    path: str,
    laser: Laser,
    name: str,
    extent: Tuple[float, float, float, float],
    viewconfig: dict,
    size: Tuple[int, int] = (1280, 800),
    include_colorbar: bool = False,
    include_scalebar: bool = False,
    include_label: bool = False,
) -> None:
    figsize = (size[0] / 100.0, size[1] / 100.0)
    fig = Figure(frameon=False, tight_layout=True, figsize=figsize, dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    data = laser.get(name, calibrate=True)
    plotLaserImage(
        fig,
        ax,
        data,
        aspect=laser.config.aspect(),
        cmap=viewconfig["cmap"]["type"],
        colorbar=include_colorbar,
        colorbarpos="bottom",
        colorbartext=str(laser.data[name].unit),
        # TODO Change this to take pixel size? Add a data+size to extent func?
        extent=(0.0, data.shape[1] * laser.config.pixel_width(), 0.0, data.shape[0] * laser.config.pixel_height()
        fontsize=viewconfig["font"]["size"],
        interpolation=viewconfig["interpolation"].lower(),
        label=include_label,
        labeltext=formatIsotope(name, fstring="$^{{{mass}}}${element}"),
        scalebar=include_scalebar,
        vmax=viewconfig["cmap"]["range"][1],
        vmin=viewconfig["cmap"]["range"][0],
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    fig.savefig(path, transparent=True, frameon=False)
    fig.clear()
    canvas.close_event()
