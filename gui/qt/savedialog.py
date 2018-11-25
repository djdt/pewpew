from PyQt5 import QtCore, QtWidgets

from util.exporter import exportNpz, exportCsv

import os.path


class SaveDialog(QtWidgets.QFileDialog):
    def __init__(self, parent=None, directory=""):
        super().__init__(parent, "Save Data", directory,
                         "Numpy archive(*.npz);;CSV(*.csv);;PNG image(*.png)")
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.setFileMode(QtWidgets.QFileDialog.AnyFile)

    def save(self, data):
        path = self.selectedFiles()[0]
        ext = os.path.splitext(path)[1]
        if ext == ".npz":
            exportNpz(path, [data])
        elif ext == ".csv":
            exportCsv(path, data)
        elif ext == ".png":
            # TODO show a config dialog
            # exportImage(self.getOpenFileNames()[0], data)
            pass
        else:
            QtWidgets.QMessageBox.warning(
                self, "Unknown Extension",
                f"Unknown file format for {os.path.basename(path)}!")
            QtWidgets.QMessageBox.critical(
                self, "Unknown Extension",
                f"Unknown file format for {os.path.basename(path)}!")
