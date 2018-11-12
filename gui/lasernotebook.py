from util.laserimage import LaserImage
from util.laser import LaserData
from gui.plotnotebook import PlotPage

import wx
from wx.lib.agw.aui import AuiNotebook


class LaserNoteBook(wx.Panel):
    def __init__(self, parent, data, params):
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        self.nb = AuiNotebook(self)
        sizer = wx.BoxSizer()
        sizer.Add(self.nb, 1, wx.EXPAND)
        self.SetSizer(sizer)

        self.data = LaserData(data, params)

    def addIsotopes(self):
        for isotope in self.data.isotopes():
            page = self.add(isotope)
            ax = page.fig.add_subplot(111)
            LaserImage(page.fig, ax, self.data.get(isotope), label=isotope,
                       aspect=self.data.aspect(), extent=self.data.extent())

    def add(self, name):
        page = PlotPage(self.nb)
        self.nb.AddPage(page, name)
        return page
