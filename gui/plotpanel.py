from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from util.laserimage import LaserImage

from util.plothelpers import coords2index

import numpy as np
import wx


class PlotPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_DOUBLE)

        self.fig = Figure(frameon=False, facecolor='black')
        self.canvas = FigureCanvas(self, wx.ID_ANY, self.fig)
        initdata = np.load('gui/image.npy')
        print(initdata)
        self.image = LaserImage(self.fig, initdata=initdata)
        self.fig.tight_layout()
        self.canvas.draw()

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

        self.canvas.callbacks.connect('motion_notify_event', self.onMotion)

    def update(self, data, label=None, aspect='auto', extent=None):
        self.image.update(data, label, aspect, extent)
        self.fig.tight_layout()
        self.canvas.draw()

    def onMotion(self, e):
        # Update status bar
        if e.inaxes:
            x, y = e.xdata, e.ydata
            ix, iy = coords2index(self.image.im, x, y)
            val = self.image.im.get_array()[ix, iy]
            self.GetParent().SetStatusText(f"({x:.0f}, {y:.0f}) [{val:.0f}]")
