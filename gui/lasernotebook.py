from util.laserimage import LaserImage
from util.laser import LaserData

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure

import wx
from wx.aui import AuiNotebook


class LaserNoteBookPage(wx.Panel):
    def __init__(self, parent, isotope):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        self.isotope = isotope

        self.fig = Figure(frameon=False, facecolor='black')
        self.canvas = FigureCanvasWxAgg(self, wx.ID_ANY, self.fig)
        sizer = wx.BoxSizer()
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 2)
        self.SetSizer(sizer)

    def draw(self, data):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        # Calibrate data
        img = (data.get(self.isotope) - data.params.intercept)\
            / data.params.gradient
        LaserImage(self.fig, ax, img, label=self.isotope,
                   aspect=data.aspect(), extent=data.extent())
        self.fig.tight_layout()
        self.canvas.draw()


class LaserNoteBook(AuiNotebook):
    def __init__(self, parent, data, params):
        AuiNotebook.__init__(self, parent, wx.ID_ANY)

        self.data = LaserData(data, params)
        self.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSED, self.onPageClose)

    def addIsotopes(self):
        for isotope in self.data.isotopes():
            page = self.add(isotope)
            page.draw(self.data)

    def add(self, isotope):
        page = LaserNoteBookPage(self, isotope)
        self.AddPage(page, isotope)
        return page

    def update(self):
        for i in range(0, self.GetPageCount()):
            page = self.GetPage(i)
            page.draw(self.data)

    def onPageClose(self, e):
        pageid = e.GetSelection()
        self.DeletePage(pageid)
        self.RemovePage(pageid)
        self.Refresh()
