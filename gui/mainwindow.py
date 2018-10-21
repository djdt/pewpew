import wx
from util.importers import AgilentImporter
from gui.plotpanel import PlotPanel


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(700, 500))
        self.CreateStatusBar()

        self.createMenuBar()
        self.createWidgets()

    def createWidgets(self):
        # Sizers and main panel
        box = wx.BoxSizer(wx.HORIZONTAL)
        boxLeft = wx.BoxSizer(wx.VERTICAL)
        gridRight = wx.GridSizer(0, 2, 0, 0)

        # Left side (image and element selector)
        self.plot = PlotPanel(self)
        boxLeft.Add(self.plot, 1, wx.ALL | wx.EXPAND | wx.GROW, 5)

        self.elementCombo = wx.ComboBox(self, wx.ID_ANY, "Elements")
        self.Bind(wx.EVT_COMBOBOX, self.onComboElements, self.elementCombo)
        boxLeft.Add(self.elementCombo, 0, wx.ALL | wx.EXPAND, 5)

        box.Add(boxLeft, 1, wx.EXPAND, 0)
        box.Add(gridRight, 0, wx.EXPAND, 0)

        self.SetSizer(box)
        self.Layout()

    def createMenuBar(self):

        menuBar = wx.MenuBar()

        # Filemenu
        fileMenu = wx.Menu()
        menuOpen = fileMenu.Append(wx.ID_OPEN, "&Open", "Open some data.")
        menuExit = fileMenu.Append(wx.ID_EXIT, "E&xit", "Quit the program.")

        menuBar.Append(fileMenu, "&File")

        # Helpmenu
        helpMenu = wx.Menu()
        menuAbout = helpMenu.Append(wx.ID_ABOUT, "&About",
                                    "About this program.")
        menuBar.Append(helpMenu, "&Help")

        # Binds
        self.Bind(wx.EVT_MENU, self.onOpen, menuOpen)
        self.Bind(wx.EVT_MENU, self.onExit, menuExit)
        self.Bind(wx.EVT_MENU, self.onAbout, menuAbout)

        self.SetMenuBar(menuBar)

    def onComboElements(self, e):
        # Update image
        data = self.layer[self.elementCombo.GetStringSelection()]
        self.plot.updateImage(data)

    def onOpen(self, e):
        dlg = wx.DirDialog(self, "Select batch directory.", "",
                           wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        dlg.ShowModal()
        batchdir = dlg.GetPath()
        if not batchdir.endswith('.b'):
            self.SetStatusTexSetStatusText("Invalid batch directory.")
        else:
            # Load the layer
            self.layer = AgilentImporter.getLayer(batchdir)
            # Update combo
            self.elementCombo.SetItems(self.layer.dtype.names)
            self.elementCombo.SetSelection(0)
            # Update image
            data = self.layer[self.elementCombo.GetStringSelection()]
            self.plot.updateImage(data)
        dlg.Destroy()

    def onExit(self, e):
        self.Close(True)

    def onAbout(self, e):
        dlg = wx.MessageDialog(self, "Image generation for LA-ICP-MS",
                               "About Laserplot", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()
