import wx
from util.laser import Laser
from gui.plotpanel import PlotPanel


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        self.laser = Laser()

        wx.Frame.__init__(self, parent, title=title, size=(700, 500))
        self.CreateStatusBar()

        self.createMenuBar()
        self.createWidgets()

    def createWidgets(self):
        # Sizers and main panel
        box = wx.BoxSizer(wx.HORIZONTAL)
        boxLeft = wx.BoxSizer(wx.VERTICAL)
        boxRight = wx.BoxSizer(wx.VERTICAL)

        # Left side (image and isotope selector)
        self.plot = PlotPanel(self)
        boxLeft.Add(self.plot, 1, wx.ALL | wx.EXPAND | wx.GROW, 5)
        self.Bind(wx.EVT_MOUSE_EVENTS, self.onMousePlot, self.plot)

        self.isotopeCombo = wx.ComboBox(self, value="Isotope")
        self.isotopeCombo.SetEditable(False)
        self.Bind(wx.EVT_COMBOBOX, self.onComboIsotope, self.isotopeCombo)
        boxLeft.Add(self.isotopeCombo, 0, wx.ALL | wx.ALIGN_RIGHT, 5)

        # Right side, inputs
        boxRight.Add(wx.StaticText(self, label="Laser parameters"),
                     0, wx.ALL | wx.CENTER, 5)

        boxRight.Add(wx.StaticLine(self), 0, wx.ALL | wx.EXPAND, 5)

        # # Grid for laser params
        gridParams = wx.GridSizer(0, 2, 0, 0)
        gridParams.Add(wx.StaticText(self, label="Scantime (s)"),
                       0, wx.ALL | wx.ALIGN_CENTER, 5)
        # TODO add validators to check input
        self.ctrlScantime = wx.TextCtrl(self, value=str(self.laser.scantime))
        gridParams.Add(self.ctrlScantime, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        gridParams.Add(wx.StaticText(self, label="Speed (μm/s)"),
                       0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.ctrlSpeed = wx.TextCtrl(self, value=str(self.laser.speed))
        gridParams.Add(self.ctrlSpeed, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        gridParams.Add(wx.StaticText(self, label="Spotsize (μm)"),
                       0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.ctrlSpotsize = wx.TextCtrl(self, value=str(self.laser.spotsize))
        gridParams.Add(self.ctrlSpotsize, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        boxRight.Add(gridParams, 0, wx.EXPAND, 5)

        boxRight.Add(wx.StaticLine(self), 0, wx.ALL | wx.EXPAND, 5)

        box.Add(boxLeft, 1, wx.EXPAND, 0)
        box.Add(boxRight, 0, wx.EXPAND, 0)
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

    def updateImage(self):
        data = self.laser.getData(self.isotopeCombo.GetStringSelection())
        self.plot.update(data, aspect=self.laser.getAspect(),
                         extent=self.laser.getExtent())
        self.plot.Refresh()

    def onMousePlot(self, e):
        pass

    def onComboIsotope(self, e):
        self.updateImage()

    def onCalibrate(self, e):
        pass

    def onOpen(self, e):
        dlg = wx.DirDialog(self, "Select batch directory.", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        dlg.ShowModal()
        batchdir = dlg.GetPath()
        if not batchdir.endswith('.b'):
            self.SetStatusText("Invalid batch directory.")
        else:
            # Load the layer
            self.laser.importData(batchdir, importer='Agilent')
            # Update combo
            self.isotopeCombo.SetItems(self.laser.getIsotopes())
            self.isotopeCombo.SetSelection(0)
            # Update image
            self.updateImage()
        dlg.Destroy()

    def onExit(self, e):
        self.Close(True)

    def onAbout(self, e):
        dlg = wx.MessageDialog(self, "Image generation for LA-ICP-MS",
                               "About Laserplot", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()
