from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.formatter import isotopeFormat


class LaserImage(object):
    def __init__(self, fig, ax, data, extent=None, aspect='auto',
                 colorbar=True, scalebar=True, label=None,
                 cmap='magma'):
        self.fig = fig
        self.ax = ax

        self.im = self.ax.imshow(data, cmap=cmap, interpolation='none',
                                 extent=extent, aspect=aspect)
        self.ax.set_axis_off()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_facecolor('black')

        if colorbar:
            self.addColorBar()
        if scalebar:
            self.addScaleBar()

        if label is not None:
            self.addLabel(label)

    def addColorBar(self):
        self.has_colorbar = True
        div = make_axes_locatable(self.ax)
        self.cax = div.append_axes('right', size=0.1, pad=0.05)
        self.fig.colorbar(self.im, cax=self.cax)

    def addScaleBar(self):
        scalebar = ScaleBar(1.0, 'um', frameon=False, color='white')
        self.ax.add_artist(scalebar)

    def addLabel(self, label):
        self.ax.annotate(isotopeFormat(label),
                         xycoords='axes fraction', xy=(0.05, 1.0),
                         textcoords='offset pixels', xytext=(0, -24),
                         ha='center',
                         color='white')
