from matplotlib.ticker import MaxNLocator
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.formatter import isotopeFormat
import numpy as np


def plotLaserImage(fig,
                   ax,
                   data,
                   interpolation=None,
                   extent=None,
                   aspect='auto',
                   colorbar=None,
                   colorbarpos='bottom',
                   colorbarlabel='',
                   scalebar=True,
                   label=None,
                   vmin='0%',
                   vmax='100%',
                   cmap='magma'):

    if type(vmin) == str:
        vmin = np.percentile(data, float(vmin.rstrip('%')))
    if type(vmax) == str:
        vmax = np.percentile(data, float(vmax.rstrip('%')))

    im = ax.imshow(
        data,
        cmap=cmap,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect=aspect)

    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_facecolor('black')
    ax.axis('scaled')

    if scalebar:
        scalebar = ScaleBar(1.0, 'um', frameon=False, color='white')
        ax.add_artist(scalebar)

    if label is not None and label is not "":
        ax.annotate(
            isotopeFormat(label),
            xycoords='axes fraction',
            xy=(0.0, 1.0),
            textcoords='offset points',
            xytext=(16, -16),
            ha='center',
            color='white')

    if colorbar is not None:
        div = make_axes_locatable(ax)
        cax = div.append_axes(colorbarpos, size=0.1, pad=0.05)
        if colorbarpos in ['right', 'left']:
            orientation = 'vertical'
        else:
            orientation = 'horizontal'
        fig.colorbar(
            im,
            label=colorbarlabel,
            cax=cax,
            orientation=orientation,
            ticks=MaxNLocator(nbins=6))

    return im
