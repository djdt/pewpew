import wx
import wx.lib.mixins.inspection as wit
from gui.mainwindow import MainWindow
from util.laser import LaserParams

# app = wx.App()
app = wit.InspectableApp()
frame = MainWindow(None, "Laserplot")
frame.nb.addBatch("/home/tom/Downloads/HER2 overnight.b", LaserParams())
frame.Show(True)
wx.lib.inspection.InspectionTool().Show()
app.MainLoop()
