from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pewpew.lib.plotimage import plot_laser_data
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
    data = laser.get(name, calibrate=True, extent=extent)
    plot_laser_data(
        fig,
        ax,
        data,
        aspect=laser.config.aspect(),
        cmap=viewconfig["cmap"]["type"],
        colorbar="bottom" if include_colorbar else None,
        colorbar_label=str(laser.data[name].unit),
        colorbar_range=viewconfig["cmap"]["range"],
        extent=extent,
        fontsize=viewconfig["font"]["size"],
        interpolation=viewconfig["interpolation"].lower(),
        label=formatIsotope(name, fstring="$^{{{mass}}}${element}"),
        scalebar=include_scalebar,
    )
    # ax.set_xlim(extent[0], extent[1])
    # ax.set_ylim(extent[2], extent[3])

    fig.savefig(path, transparent=True, frameon=False)
    fig.clear()
    canvas.close_event()
