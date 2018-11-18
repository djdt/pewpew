import wx


class AuiNBClose(wx.aui.AuiNotebook):
    def __init__(self, parent, id=wx.ID_ANY):
        wx.aui.AuiNotebook.__init__(self, parent, id)

        self.Bind(wx.aui.EVT_AUINOTEBOOK_PAGE_CLOSED, self.onPageClose)

    def onPageClose(self, e):
        pageid = e.GetSelection()
        self.DeletePage(pageid)
