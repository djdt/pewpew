import wx

from util.laser import LaserParams


class LaserControls(wx.Panel):

    def __init__(self, parent, params=LaserParams()):
        wx.Panel.__init__(self, parent, wx.ID_ANY)

        box = wx.BoxSizer(wx.VERTICAL)

        box.Add(wx.StaticText(self, label="Laser parameters"),
                0, wx.ALL | wx.CENTER, 5)

        box.Add(wx.StaticLine(self), 0, wx.ALL | wx.EXPAND, 5)

        # Grid for laser params
        grid = wx.GridSizer(0, 2, 0, 0)
        grid.Add(wx.StaticText(self, label="Scantime (s)"),
                 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.ctrlScantime = wx.TextCtrl(self, value=str(params.scantime))
        grid.Add(self.ctrlScantime, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        grid.Add(wx.StaticText(self, label="Speed (μm/s)"),
                 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.ctrlSpeed = wx.TextCtrl(self, value=str(params.speed))
        grid.Add(self.ctrlSpeed, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        grid.Add(wx.StaticText(self, label="Spotsize (μm)"),
                 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.ctrlSpotsize = wx.TextCtrl(self, value=str(params.spotsize))
        grid.Add(self.ctrlSpotsize, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        box.Add(grid, 0, wx.EXPAND, 5)

        box.Add(wx.StaticLine(self), 0, wx.ALL | wx.EXPAND, 5)

        # Grid for calibration
        grid = wx.GridSizer(0, 2, 0, 0)
        grid.Add(wx.StaticText(self, label="Gradient"),
                 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.ctrlGradient = wx.TextCtrl(self, value=str(params.gradient))
        grid.Add(self.ctrlGradient, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        grid.Add(wx.StaticText(self, label="Intercept"),
                 0, wx.ALL | wx.ALIGN_CENTER, 5)
        self.ctrlIntercept = wx.TextCtrl(self, value=str(params.intercept))
        grid.Add(self.ctrlIntercept, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        box.Add(grid, 0, wx.EXPAND, 5)

        self.button = wx.Button(self, label="Update")
        box.Add(self.button, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        box.Add(wx.StaticLine(self), 0, wx.ALL | wx.EXPAND, 5)

        self.SetSizer(box)

    def updateParams(self, params):
        self.ctrlScantime.ChangeValue(str(params.scantime))
        self.ctrlSpeed.ChangeValue(str(params.speed))
        self.ctrlSpotsize.ChangeValue(str(params.spotsize))
        self.ctrlGradient.ChangeValue(str(params.gradient))
        self.ctrlIntercept.ChangeValue(str(params.intercept))

    def getParams(self):
        params = LaserParams(scantime=float(self.ctrlScantime.GetValue()),
                             speed=float(self.ctrlSpeed.GetValue()),
                             spotsize=float(self.ctrlSpotsize.GetValue()),
                             gradient=float(self.ctrlGradient.GetValue()),
                             intercept=float(self.ctrlIntercept.GetValue()))
        return params
