import wx
from gui.mainwindow import MainWindow

app = wx.App()
frame = MainWindow(None, "Laserplot")
frame.Show(True)
app.MainLoop()
