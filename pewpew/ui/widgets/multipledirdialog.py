from PyQt5 import QtWidgets

from typing import List


class MultipleDirDialog(QtWidgets.QFileDialog):
    def __init__(self, title: str, directory: str, parent: QtWidgets.QWidget = None):
        super().__init__(parent, title, directory)
        self.setFileMode(QtWidgets.QFileDialog.Directory)
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        for view in self.findChildren((QtWidgets.QListView, QtWidgets.QTreeView)):
            if isinstance(view.model(), QtWidgets.QFileSystemModel):
                view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

    @staticmethod
    def getExistingDirectories(
        parent: QtWidgets.QWidget, title: str, directory: str
    ) -> List[str]:
        dlg = MultipleDirDialog(title, directory, parent)
        if dlg.exec():
            return list(dlg.selectedFiles())
        else:
            return []
