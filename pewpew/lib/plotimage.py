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
    aspect: Union[str, float] = "auto",
    cmap: str = "magma",
    colorbar: bool = False,
    colorbarpos: str = "bottom",
    colorbartext: str = "",
    extent: Tuple[float, float, float, float] = None,
    fontsize: int = 12,
    interpolation: str = "none",
    label: bool = False,
    labeltext: str = "",
    scalebar: bool = False,
    xaxis: bool = False,
    xaxisticksize: float = None,
    vmax: Union[str, float] = "100%",
    vmin: Union[str, float] = "0%",
) -> AxesImage:

    # If data is empty create a dummy data
    if data is None or data.size == 0:
        data = np.array([[0]], dtype=np.float64)

    # Calculate the colorbar range
    if isinstance(vmin, str):
        vmin = np.percentile(data, float(vmin.rstrip("%")))
    if isinstance(vmax, str):
        vmax = np.percentile(data, float(vmax.rstrip("%")))

    # Plot the image
    im = ax.imshow(
        data,
        cmap=cmap,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect=aspect,
    )

    if colorbar:
        div = make_axes_locatable(ax)
        cax = div.append_axes(colorbarpos, size=0.1, pad=0.05)
        if colorbarpos in ["right", "left"]:
            orientation = "vertical"
        else:
            orientation = "horizontal"
        fig.colorbar(
            im,
            label=colorbartext,
            cax=cax,
            orientation=orientation,
            ticks=MaxNLocator(nbins=6),
        )

    if label:
        text = AnchoredText(
            labeltext,
            "upper left",
            pad=0.2,
            borderpad=0.1,
            frameon=False,
            prop={"color": "white", "size": fontsize},
        )
        ax.add_artist(text)

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

    ax.axis("scaled")
    ax.set_facecolor("black")
    if xaxis:
        ax.tick_params(axis='x', direction='in', color='white', labelbottom=False)
        if xaxisticksize is not None:
            start = extent[0] if extent is not None else 0.0
            start += start % xaxisticksize
            end = extent[1] if extent is not None else data.shape[1]
            xticks = np.arange(start, end, xaxisticksize)
            ax.set_xticks(xticks)
    else:
        ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return im
