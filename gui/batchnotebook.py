import os.path

import wx
from wx.aui import AuiNotebook

import util.importers
from gui.lasernotebook import LaserNoteBook


class BatchNotebook(AuiNotebook):
    def __init__(self, parent):
        AuiNotebook.__init__(self, parent, wx.ID_ANY)
        # self.nb = AuiNotebook(self)
        # sizer = wx.BoxSizer()
        # sizer.Add(self, 1, wx.EXPAND)
        # self.SetSizer(sizer)

    def add(self, name, data, params):
        page = LaserNoteBook(self, data, params)
        self.AddPage(page, name)
        return page

    def addBatch(self, path, params, importer='agilent'):
        if importer == 'agilent':
            data = util.importers.importAgilentBatch(path)
            page = self.add(os.path.basename(path), data, params)
            page.addIsotopes()
        else:
            wx.MessageDialog(self, f"Unknown importer {importer}!")
