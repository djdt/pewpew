import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.plothelpers import coords2index
from util.formatter import isotopeFormat

import wx
import numpy as np


class PlotPanel(wx.Panel):
    def __init__(self, parent, size=(400, 400), cmap='magma'):
        wx.Panel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_DOUBLE)

        self.cmap = cmap

        self.fig = matplotlib.figure.Figure(
                (size[0] / 100.0, size[1] / 100.0),
                dpi=100, frameon=False, facecolor='black')
        self.ax = self.fig.add_subplot(111)
        div = make_axes_locatable(self.ax)
        self.cax = div.append_axes('right', size='5%', pad=0.05)
        self.canvas = FigureCanvas(self, wx.ID_ANY, self.fig)

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

        self._init_plot()

        self.canvas.callbacks.connect('motion_notify_event', self.onMotion)

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_facecolor('black')
        data = np.load('gui/image.npy')
        self.im = self.ax.imshow(data, cmap=self.cmap,
                                 interpolation='none')
        # Draw colorbar
        self.cb = self.fig.colorbar(self.im, cax=self.cax)

        self.fig.tight_layout()
        self.canvas.draw()

    def update(self, data, label=None, aspect='auto', extent=None):
        self.ax.clear()
        # Plot the image
        self.im = self.ax.imshow(data, cmap=self.cmap,
                                 interpolation='none',
                                 extent=extent, aspect=aspect)
        # Create / update colorbarc
        self.cb.update_normal(self.im)
        # Draw scalebar
        scalebar = ScaleBar(1.0, 'um', frameon=False, color='white')
        self.ax.add_artist(scalebar)
        # Draw isotope label
        if label is not None:
            self.ax.annotate(isotopeFormat(label),
                             xycoords='axes fraction', xy=(0.0, 0.85),
                             textcoords='offset pixels', xytext=(0, -5),
                             color='white')

        self.fig.tight_layout()
        self.canvas.draw()

    def onMotion(self, e):
        # Update status bar
        if e.inaxes:
            x, y = e.xdata, e.ydata
            ix, iy = coords2index(self.im, x, y)
            val = self.im.get_array()[ix, iy]
            self.GetParent().SetStatusText(f"({x:.0f}, {y:.0f}) [{val:.0f}]")
