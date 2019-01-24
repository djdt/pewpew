import os.path

from PyQt5 import QtWidgets

from gui.docks.image import ImageDock

from util.exporter import exportCsv, exportNpz, exportPng
from util.laser import LaserData


class LaserImageDock(ImageDock):
    def __init__(self, laserdata: LaserData, parent: QtWidgets.QWidget = None):

        super().__init__(parent)
        self.laser = laserdata
        self.combo_isotope.addItems(self.laser.isotopes())
        self.setWindowTitle(self.laser.name)

    def _export(
        self,
        path: str,
        isotope: str = None,
        layer: int = None,
        prompt_overwrite: bool = True,
    ) -> QtWidgets.QMessageBox.StandardButton:
        if isotope is None:
            isotope = self.combo_isotope.currentText()

        result = QtWidgets.QMessageBox.Yes
        if prompt_overwrite and os.path.exists(path):
            result = QtWidgets.QMessageBox.warning(
                self,
                "Overwrite File?",
                f'The file "{os.path.basename(path)}" '
                "already exists. Do you wish to overwrite it?",
                QtWidgets.QMessageBox.Yes
                | QtWidgets.QMessageBox.YesToAll
                | QtWidgets.QMessageBox.No,
            )
            if result == QtWidgets.QMessageBox.No:
                return result

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            exportCsv(
                path,
                self.laser.get(isotope, calibrated=True, trimmed=True),
                isotope,
                self.laser.config,
            )
        elif ext == ".npz":
            exportNpz(path, [self.laser])
        elif ext == ".png":
            exportPng(
                path,
                self.laser.get(isotope, calibrated=True, trimmed=True),
                isotope,
                self.laser.aspect(),
                self.laser.extent(trimmed=True),
                self.window().viewconfig,
            )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Format",
                f"Unknown extention for '{os.path.basename(path)}'.",
            )
            return QtWidgets.QMessageBox.NoToAll

        return result
