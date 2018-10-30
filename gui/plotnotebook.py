import wx
import wx.lib.agw.aui as aui
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg


class PlotPage(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, wx.ID_ANY)

        self.fig = Figure(frameon=False, facecolor='black')
        self.canvas = FigureCanvasWxAgg(self, wx.ID_ANY, self.fig)
        sizer = wx.BoxSizer()
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)


class PlotNoteBook(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        self.nb = aui.AuiNotebook(self)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def add(self, name):
        page = PlotPage(self.nb)
        self.nb.AddPage(page, name)
        return page.fig
