from util.laserimage import LaserImage
from util.laser import LaserData
from util.plothelpers import coords2value

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure

import wx
from wx.aui import AuiNotebook


class LaserNoteBookPage(wx.Panel):
    def __init__(self, parent, isotope):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        self.isotope = isotope

        self.fig = Figure(frameon=False, facecolor='black')
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasWxAgg(self, wx.ID_ANY, self.fig)
        sizer = wx.BoxSizer()
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 2)
        self.SetSizer(sizer)

        # Mouse pos events
        self.canvas.mpl_connect('motion_notify_event', self.updateStatusBar)
        self.canvas.mpl_connect('axes_leave_event', self.clearStatusBar)

    def draw(self, data):
        self.ax.clear()
        # Calibrate data
        img = (data.get(self.isotope) - data.params.intercept)\
            / data.params.gradient
        self.lase = LaserImage(self.fig, self.ax, img, label=self.isotope,
                               aspect=data.aspect(), extent=data.extent())
        self.fig.tight_layout()
        self.canvas.draw()

    def updateStatusBar(self, e):
        # TODO make sure no in the color bar axes
        if e.inaxes:
            x, y = e.xdata, e.ydata
            v = coords2value(self.lase.im, x, y)
            self.GetTopLevelParent().SetStatusText(
                f"{x:.2f},{y:.2f} [{v}]")

    def clearStatusBar(self, e):
        self.GetTopLevelParent().SetStatusText("")


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
        self.Refresh()

    def onPageClose(self, e):
        pageid = e.GetSelection()
        self.DeletePage(pageid)
        self.RemovePage(pageid)
        self.Refresh()
