from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pewpew.lib.plotimage import plotLaserImage
from pewpew.lib.laser import LaserData
from pewpew.lib.formatter import formatIsotope

from typing import Tuple


def save(
    path: str,
    laser: LaserData,
    isotope: str,
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
    plotLaserImage(
        fig,
        ax,
        laser.get(isotope, calibrated=True),
        aspect=laser.aspect(),
        cmap=viewconfig["cmap"]["type"],
        colorbar=include_colorbar,
        colorbarpos="bottom",
        colorbartext=str(laser.calibration[isotope]["unit"]),
        extent=laser.extent(),
        fontsize=viewconfig["font"]["size"],
        interpolation=viewconfig["interpolation"].lower(),
        label=include_label,
        labeltext=formatIsotope(isotope, fstring="$^{{{mass}}}${element}"),
        scalebar=include_scalebar,
        vmax=viewconfig["cmap"]["range"][1],
        vmin=viewconfig["cmap"]["range"][0],
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    fig.savefig(path, transparent=True, frameon=False)
    fig.clear()
    canvas.close_event()
