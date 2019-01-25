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
    viewconfig: dict,
    size: Tuple[int, int] = (640, 480),
    include_colorbar: bool = False,
    include_scalebar: bool = False,
    include_label: bool = False,
) -> None:
    figsize = (size[1] / 100.0, size[0] / 100.0)
    fig = Figure(frameon=False, tight_layout=True, figsize=figsize, dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    plotLaserImage(
        fig,
        ax,
        laser.get(isotope, calibrated=True, trimmed=True),
        aspect=laser.aspect(),
        cmap=viewconfig["cmap"]["type"],
        colorbar=include_colorbar,
        colorbarpos="bottom",
        colorbartext=str(laser.calibration[isotope]["unit"]),
        extent=laser.extent(trimmed=True),
        fontsize=viewconfig["font"]["size"],
        interpolation=viewconfig["interpolation"].lower(),
        label=include_label,
        labeltext=formatIsotope(isotope, fstring="$^{{{mass}}}${element}"),
        scalebar=include_scalebar,
        vmax=viewconfig["cmap"]["range"][1],
        vmin=viewconfig["cmap"]["range"][0],
    )
    fig.savefig(path, transparent=True, frameon=False)
    fig.clear()
    canvas.close_event()
