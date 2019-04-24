import numpy as np

from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import Tuple, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def plot_laser_data(
    fig: Figure,
    ax: Axes,
    data: np.ndarray,
    aspect: Union[str, float] = "auto",
    cmap: str = "magma",
    extent: Tuple[float, float, float, float] = None,
    interpolation: str = "none",
    alpha: float = 1.0,
    colorbar: str = None,
    colorbar_range: Tuple[Union[str, float], Union[str, float]] = ("1%", "99%"),
    colorbar_label: str = "",
    label: str = None,
    scalebar: str = None,
    fontsize: int = 12,
) -> AxesImage:

    # If data is empty create a dummy data
    if data is None or data.size == 0:
        data = np.array([[0]], dtype=np.float64)

    # Calculate the colorbar range
    if isinstance(colorbar_range[0], str):
        vmin = np.percentile(data, float(colorbar_range[0].rstrip("%")))
    else:
        vmin = float(colorbar_range[0])
    if isinstance(colorbar_range[1], str):
        vmax = np.percentile(data, float(colorbar_range[1].rstrip("%")))
    else:
        vmax = float(colorbar_range[1])

    # Plot the image
    im = ax.imshow(
        data,
        cmap=cmap,
        interpolation=interpolation,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect=aspect,
        origin="upper",
    )

    if colorbar is not None:
        div = make_axes_locatable(ax)
        cax = div.append_axes(colorbar, size=0.1, pad=0.05)
        if colorbar in ["right", "left"]:
            orientation = "vertical"
        else:
            orientation = "horizontal"
        fig.colorbar(
            im,
            label=colorbar_label,
            cax=cax,
            orientation=orientation,
            ticks=MaxNLocator(nbins=6),
        )

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

    if scalebar is not None:
        scalebar = ScaleBar(
            1.0,
            "um",
            location=scalebar,
            frameon=False,
            color="white",
            font_properties={"size": fontsize},
        )
        ax.add_artist(scalebar)

    ax.axis("scaled")
    ax.set_facecolor("black")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return im
