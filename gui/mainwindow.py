import wx
import wx.aui
from util.laser import LaserParams
# from gui.plotpanel import PlotPanel
# from gui.lasernotebook import LaserNoteBook
from gui.batchnotebook import BatchNotebook
from gui.controls import LaserControls


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        # self.laser = Laser()
        self.params = LaserParams()

        wx.Frame.__init__(self, parent, title=title, size=(1280, 800))
        self.CreateStatusBar()

        self.createMenuBar()
        self.createWidgets()

    def createWidgets(self):
        # Sizers and main panel
        box = wx.BoxSizer(wx.HORIZONTAL)
        # boxLeft = wx.BoxSizer(wx.VERTICAL)

        # Left side (image and isotope selector)
        # self.plot = PlotPanel(self)
        self.nb = BatchNotebook(self)
        self.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CHANGED,
                  self.onNoteBookChanged, self.nb)

        self.controls = LaserControls(self, self.params)
        self.Bind(wx.EVT_BUTTON, self.onControlUpdate, self.controls.button)

        box.Add(self.nb, 1, wx.EXPAND, 5)
        box.Add(self.controls, 0, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(box)
        self.Layout()

    def createMenuBar(self):

        menuBar = wx.MenuBar()

        # Filemenu
        fileMenu = wx.Menu()
        menuOpen = fileMenu.Append(wx.ID_OPEN, "&Open", "Open some data.")
        menuExit = fileMenu.Append(wx.ID_EXIT, "E&xit", "Quit the program.")

        self.Bind(wx.EVT_MENU, self.onOpen, menuOpen)
        self.Bind(wx.EVT_MENU, self.onExit, menuExit)

        menuBar.Append(fileMenu, "&File")

        # Editmenu
        editMenu = wx.Menu()
        menuCalibrate = editMenu.Append(wx.ID_INFO, "&Calibration",
                                        "Set calibration parameters.")
        self.Bind(wx.EVT_MENU, self.onCalibrate, menuCalibrate)

        menuBar.Append(editMenu, "&Edit")

        # Helpmenu
        helpMenu = wx.Menu()
        menuAbout = helpMenu.Append(wx.ID_ABOUT, "&About",
                                    "About this program.")
        self.Bind(wx.EVT_MENU, self.onAbout, menuAbout)

        menuBar.Append(helpMenu, "&Help")

        # Binds

        self.SetMenuBar(menuBar)

    # def updateImage(self):
    #     isotope = self.isotopeCombo.GetStringSelection()
    #     data = self.laser.getData(isotope)
    #     self.plot.update(data, label=isotope, aspect=self.laser.getAspect(),
    #                      extent=self.laser.getExtent())
    #     self.plot.Refresh()

    def onMousePlot(self, e):
        pass

    # Menu Events
    # def onComboIsotope(self, e):
    #     self.updateImage()

    def onCalibrate(self, e):
        pass

    def onControlUpdate(self, e):
        page = self.nb.GetCurrentPage()
        params = self.controls.getParams()
        page.data.params = params

        page.update()

    def onNoteBookChanged(self, e):
        params = self.nb.GetCurrentPage().data.params
        self.controls.updateParams(params)

    def onOpen(self, e):
        dlg = wx.DirDialog(self, "Select batch directory.", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        dlg.ShowModal()
        batch = dlg.GetPath()
        if not batch.endswith('.b'):
            self.SetStatusText("Invalid batch directory.")
        else:
            # Load the layer
            self.nb.addBatch(batch, importer='agilent', params=self.params)
            # laser = Laser()
            # laser.importData(batchdir, importer='agilent')
            # self.laser.importData(batchdir, importer='Agilent')
            # for i in laser.getIsotopes():
            #     fig = self.plotnb.add(i)
            #     LaserImage(fig, fig.gca(), laser.getData(i))
            # Update combo
            # self.isotopeCombo.SetItems(self.laser.getIsotopes())
            # self.isotopeCombo.SetSelection(0)
            # Update image
            # self.updateImage()
        dlg.Destroy()

    def onExit(self, e):
        self.Close(True)

    def onAbout(self, e):
        dlg = wx.MessageDialog(self, "Image generation for LA-ICP-MS",
                               "About Laserplot", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()
