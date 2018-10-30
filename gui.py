import wx
from gui.mainwindow import MainWindow

app = wx.App()
frame = MainWindow(None, "Laserplot")
frame.Show(True)
frame.nb.addBatch("/home/tom/Downloads/HER2 overnight.b")
app.MainLoop()
