from matplotlib.ticker import MaxNLocator
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.formatter import isotopeFormat
import numpy as np


class LaserImage(object):
    def __init__(self, fig, ax, data, extent=None, aspect='auto',
                 colorbar='right', colorbarlabel='',
                 scalebar=True, label=None,
                 vmin='auto', vmax='auto', cmap='magma'):

        if vmin is 'auto':
            vmin = np.percentile(data, 1)
        if vmax is 'auto':
            vmax = np.percentile(data, 99)

        self.im = ax.imshow(data, cmap=cmap, interpolation='none',
                            vmin=vmin, vmax=vmax,
                            extent=extent, aspect=aspect)
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_facecolor('black')

        if colorbar not in ['none', None]:
            self.addColorBar(fig, ax, colorbar, colorbarlabel)
        if scalebar:
            self.addScaleBar(ax)

        if label is not None:
            self.addLabel(label)

    def addColorBar(self, fig, ax, pos, label):
        div = make_axes_locatable(ax)
        cax = div.append_axes(pos, size=0.1, pad=0.05)
        if pos in ['right', 'left']:
            orientation = 'vertical'
        else:
            orientation = 'horizontal'
        fig.colorbar(self.im, label=label,
                     cax=cax, orientation=orientation,
                     ticks=MaxNLocator(nbins=6))

    def addScaleBar(self, ax):
        scalebar = ScaleBar(1.0, 'um', frameon=False, color='white')
        ax.add_artist(scalebar)

    def addLabel(self, ax, label):
        ax.annotate(isotopeFormat(label),
                    xycoords='axes fraction', xy=(0.0, 1.0),
                    textcoords='offset points', xytext=(16, -16),
                    ha='center',
                    color='white')
