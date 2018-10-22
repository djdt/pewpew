import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import numpy as np
import wx


class PlotPanel(wx.Panel):
    def __init__(self, parent, size=(400, 400), cmap='magma'):
        wx.Panel.__init__(self, parent, wx.ID_ANY)

        self.cmap = cmap

        self.fig = matplotlib.figure.Figure(
                (size[0] / 100.0, size[1] / 100.0), dpi=100, frameon=False)
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self, wx.ID_ANY, self.fig)

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

    def initImage(self):
        self.axes.imshow([[0]], cmap=self.cmap)
        self.axes.set_axis_off()
        self.canvas.draw()

    def updateImage(self, data, extent=None):
        self.axes.clear()
        self.axes.imshow(data, cmap=self.cmap, interpolation='none',
                         extent=extent)
        self.fig.tight_layout()
        self.canvas.draw()
        self.Refresh()
