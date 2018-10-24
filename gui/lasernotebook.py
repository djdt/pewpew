from gui.plotnotebook import PlotPage, PlotNoteBook
from util.laserimage import LaserImage


class LaserPage(PlotPage):
    def __init__(self, parent, data):
        PlotPage.__init__(self, parent)
        # self.fig = Figure(frameon=False, facecolor='black')
        # self.canvas = FigureCanvasWxAgg(self, wx.ID_ANY, self.fig)
        # sizer = wx.BoxSizer()
        # sizer.Add(self.nb, 1, wx.EXPAND)
        # self.SetSizer(sizer)
        ax = self.fig.add_subplot(111)
        LaserImage(self.fig, ax, data)

    def update(self, data):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        LaserImage(self.fig, ax, data)


class LaserNoteBook(PlotNoteBook):
    def __init__(self, parent):
        PlotNoteBook.__init__(self, parent)
        # self.nb = wx.aui.AuiNotebook(self)
        # sizer = wx.BoxSizer()
        # sizer.Add(self.nb, 1, wx.EXPAND)
        # self.SetSizer(sizer)

    def add(self, name, data, params):
        page = LaserPage(self.nb, data)
        self.nb.AddPage(page, name)
