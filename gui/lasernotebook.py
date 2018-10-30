from gui.plotnotebook import PlotPage, PlotNoteBook
from util.laserimage import LaserImage

from util.importers import importAgilentBatch
from util.laser import LaserParams
import os.path

# Noteception, batch -> isotopes


class LaserPage(PlotPage):
    def __init__(self, parent, data, isotope):
        PlotPage.__init__(self, parent)
        # self.fig = Figure(frameon=False, facecolor='black')
        # self.canvas = FigureCanvasWxAgg(self, wx.ID_ANY, self.fig)
        # sizer = wx.BoxSizer()
        # sizer.Add(self.nb, 1, wx.EXPAND)
        # self.SetSizer(sizer)
        ax = self.fig.add_subplot(111)
        LaserImage(self.fig, ax, data[label], label=label,
                   aspect=data.aspect(),
                   extent=data.extent())
        self.fig.tight_layout()
        self.canvas.draw()

#     def update(self, data):
#         self.fig.clear()
#         ax = self.fig.add_subplot(111)
#         LaserImage(self.fig, ax, data)


class LaserNoteBook(PlotNoteBook):
    def __init__(self, parent, params=LaserParams()):
        PlotNoteBook.__init__(self, parent)
        self.params = params
        # self.nb = wx.aui.AuiNotebook(self)
        # sizer = wx.BoxSizer()
        # sizer.Add(self.nb, 1, wx.EXPAND)
        # self.SetSizer(sizer)

    def add(self, name, label, data):
        page = LaserPage(self.nb, data, label, self.params)
        self.nb.AddPage(page, name)
        return page.fig

    def addBatch(self, batch, importer='agilent'):
        if importer is 'agilent':
            data = importAgilentBatch(batch)
        else:
            print(f'LaserNoteBook.addBatch: unknown importer fpr {batch}!')

        for i in data.dtype.names:
            name = f'{os.path.basename(batch)}:{i}'
            page = LaserPage(self.nb, data, label, self.params)
            self.nb.AddPage(page, name)
