import numpy as np

from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import Tuple, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def plotLaserImage(
    fig: Figure,
    ax: Axes,
    data: np.ndarray,
    interpolation: str = "none",
    extent: Tuple[int, int, int, int] = None,
    aspect: str = "auto",
    colorbar: bool = False,
    colorbarpos: str = "bottom",
    colorbarlabel: str = None,
    scalebar: bool = True,
    label: str = None,
    fontsize: int = 12,
    vmin: Union[str, int] = "0%",
    vmax: Union[str, int] = "100%",
    cmap: str = "magma",
) -> AxesImage:

    if data.size == 0:
        data = np.array([[0]], dtype=np.float64)

    if isinstance(vmin, str):
        vmin = np.percentile(data, float(vmin.rstrip("%")))
    if isinstance(vmax, str):
        vmax = np.percentile(data, float(vmax.rstrip("%")))

    im = ax.imshow(
        data,
        cmap=cmap,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect=aspect,
    )

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_facecolor("black")
    ax.axis("scaled")

    if scalebar:
        scalebar = ScaleBar(
            1.0,
            "um",
            location="upper right",
            frameon=False,
            color="white",
            font_properties={"size": fontsize},
        )
        ax.add_artist(scalebar)

    if label is not None:
        text = AnchoredText(
            label,
            "upper left",
            pad=0.2,
            borderpad=0.1,
            frameon=False,
            prop={"color": "white", "size": fontsize},
        )
        ax.add_artist(text)

    if colorbar:
        div = make_axes_locatable(ax)
        cax = div.append_axes(colorbarpos, size=0.1, pad=0.05)
        if colorbarpos in ["right", "left"]:
            orientation = "vertical"
        else:
            orientation = "horizontal"
        fig.colorbar(
            im,
            label=colorbarlabel,
            cax=cax,
            orientation=orientation,
            ticks=MaxNLocator(nbins=6),
        )

    return im
