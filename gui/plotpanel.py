import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib_scalebar.scalebar import ScaleBar
import wx


class PlotPanel(wx.Panel):
    def __init__(self, parent, size=(400, 400), cmap='magma'):
        wx.Panel.__init__(self, parent, wx.ID_ANY, style=wx.BORDER_DOUBLE)

        self.cmap = cmap

        self.fig = matplotlib.figure.Figure(
                (size[0] / 100.0, size[1] / 100.0),
                dpi=100, frameon=False, facecolor='black')
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self, wx.ID_ANY, self.fig)

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

        self._init_plot()

    def _init_plot(self):
        self.axes.imshow([[0]], cmap=self.cmap)
        self.axes.set_axis_off()
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        self.axes.set_facecolor('black')
        self.canvas.draw()

    def update(self, data, aspect='auto', extent=None):
        self.axes.clear()
        self.axes.imshow(data, cmap=self.cmap, interpolation='none',
                         extent=extent, aspect=aspect)
        self.axes.set_axis_off()

        scalebar = ScaleBar(1.0, 'um', frameon=False, color='white')
        self.axes.add_artist(scalebar)

        self.fig.tight_layout()
        self.canvas.draw()

    def onMotion(self, e):
        point = wx.GetMousePosition()
        self.GetParent().SetStatusText(str(point.x))
