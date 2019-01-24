import os.path

from PyQt5 import QtWidgets

from gui.docks.laserimage import LaserImageDock

from util.exporter import exportCsv, exportNpz, exportPng, exportVtr
from util.krisskross import KrissKrossData


class KrissKrossImageDock(LaserImageDock):
    def __init__(self, kkdata: KrissKrossData, parent: QtWidgets.QWidget = None):

        super().__init__(kkdata, parent)
        self.setWindowTitle(f"kk:{self.laser.name}")

        # Config cannot be changed for krisskross images
        self.action_config.setEnabled(False)
        self.action_trim.setEnabled(False)

    def onMenuConfig(self) -> None:
        pass

    def onMenuTrim(self) -> None:
        pass

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

        if layer is None:
            export_data = self.laser.get(isotope, calibrated=True, flattened=True)
        else:
            export_data = self.laser.get(isotope, calibrated=True, flattened=False)[
                :, :, layer - 1
            ]

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            exportCsv(path, export_data, isotope, self.laser.config)
        elif ext == ".npz":
            exportNpz(path, [self.laser])
        elif ext == ".png":
            exportPng(
                path,
                export_data,
                isotope,
                self.laser.aspect(),
                self.laser.extent(trimmed=True),
                self.window().viewconfig,
            )
        elif ext == ".vtr":
            exportVtr(path, self.laser)
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Format",
                f'Unknown extention for "{os.path.basename(path)}".',
            )
            return QtWidgets.QMessageBox.NoToAll

        return result
