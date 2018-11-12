import wx
from gui.mainwindow import MainWindow
from util.laser import LaserParams

app = wx.App()
frame = MainWindow(None, "Laserplot")
frame.Show(True)
frame.nb.addBatch("/home/tom/Downloads/M1 LUNG 100.b", LaserParams())
app.MainLoop()
