from pewlib.config import SpotConfig
from importlib.metadata import version
import time
import logging
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
from pewlib.io.nu import (
    is_nu_image_directory,
    read_laser_image,
    sync_data_with_laser_info,
)
from pewlib.laser import Laser
from PySide6 import QtCore, QtWidgets

from pewpew.widgets.periodictable import PeriodicTableSelector, isotope_data
from pewpew.widgets.wizards.options import search_sorted_closest

logger = logging.getLogger(__name__)


class LaserImagePathsPage(QtWidgets.QWizardPage):
    pathsChanged = QtCore.Signal()
    dataChanged = QtCore.Signal()
    massesChanged = QtCore.Signal()
    laserInfosChanged = QtCore.Signal()
    positionsChanged = QtCore.Signal()

    def __init__(self, paths: list[Path], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setTitle("Import Images")

        self._laser_datas: list[np.ndarray] = []
        self._laser_masses: list[np.ndarray] = []
        self._laser_infos: list[dict] = []
        self._laser_positions: list[QtCore.QPointF] = []

        self.image_list = QtWidgets.QListWidget()

        self.setPaths(paths)

        self.image_list.itemChanged.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_list, 0)
        self.setLayout(layout)

        self.registerField("paths", self, "paths_prop")
        self.registerField("laserdata", self, "data_prop")
        self.registerField("masses", self, "mass_prop")
        self.registerField("laserinfo", self, "info_prop")
        self.registerField("positions", self, "pos_prop")

    def isComplete(self) -> bool:
        for i in range(self.image_list.count()):
            if self.image_list.item(i).checkState() == QtCore.Qt.CheckState.Checked:
                return True

        return False

    def validatePage(self) -> bool:
        self._laser_datas.clear()
        self._laser_masses.clear()
        self._laser_infos.clear()
        self._laser_positions.clear()

        for path in self.getPaths():
            signals, masses, times, pulses, info = read_laser_image(path)
            image, pos = sync_data_with_laser_info(signals, times, pulses, info)
            self._laser_datas.append(image)
            self._laser_masses.append(masses)
            self._laser_infos.append(info)
            self._laser_positions.append(QtCore.QPointF(pos[0], pos[1]))

        return len(self._laser_datas) > 0

    def getPaths(self) -> list[Path]:
        items = [self.image_list.item(i) for i in range(self.image_list.count())]
        return [
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in items
            if item.checkState() == QtCore.Qt.CheckState.Checked
        ]

    def setPaths(self, paths: list[Path]):
        self.image_list.clear()
        for path in paths:
            if is_nu_image_directory(path):
                item = QtWidgets.QListWidgetItem()
                item.setText(str(path))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                self.image_list.addItem(item)
            else:  # catch root folders
                for dir in path.iterdir():
                    if is_nu_image_directory(dir):
                        item = QtWidgets.QListWidgetItem()
                        item.setText(str(dir))
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, dir)
                        item.setCheckState(QtCore.Qt.CheckState.Checked)
                        self.image_list.addItem(item)
        self.pathsChanged.emit()

    def getData(self) -> list[np.ndarray]:
        return self._laser_datas

    def setData(self, datas: list[np.ndarray]) -> None:
        self._laser_datas = datas
        self.dataChanged.emit()

    def getMasses(self) -> list[np.ndarray]:
        return self._laser_masses

    def setMasses(self, masses: list[np.ndarray]) -> None:
        self._laser_masses = masses
        self.massesChanged.emit()

    def getInfo(self) -> list[dict]:
        return self._laser_infos

    def setInfo(self, infos: list[dict]) -> None:
        self._laser_infos = infos
        self.laserInfosChanged.emit()

    def getPositions(self) -> list[QtCore.QPointF]:
        return self._laser_positions

    def setPositions(self, positions: list[QtCore.QPointF]) -> None:
        self._laser_pos = positions
        self.laserInfosChanged.emit()

    paths_prop = QtCore.Property("QVariant", getPaths, setPaths, notify=pathsChanged)  # type: ignore
    data_prop = QtCore.Property("QVariant", getData, setData, notify=dataChanged)  # type: ignore
    mass_prop = QtCore.Property("QVariant", getMasses, setMasses, notify=massesChanged)  # type: ignore
    info_prop = QtCore.Property("QVariant", getInfo, setInfo, notify=laserInfosChanged)  # type: ignore
    pos_prop = QtCore.Property(
        "QVariant",  # type: ignore
        getPositions,
        setPositions,
        notify=positionsChanged,
    )


class IsotopeSelectionPage(QtWidgets.QWizardPage):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        self.setTitle("Select Isotopes")
        self.table = PeriodicTableSelector()
        self.table.isotopesChanged.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table, 1)
        self.setLayout(layout)

        self.registerField(
            "selectedIsotopes", self.table, "isotopes", "isotopesChanged"
        )

    def isComplete(self) -> bool:
        return self.table.selectedIsotopes() is not None

    def initializePage(self):
        all_masses = self.field("masses")

        idx = search_sorted_closest(all_masses[0], isotope_data["mass"])
        isotopes = isotope_data[
            np.abs(all_masses[0][idx] - isotope_data["mass"]) < 0.05
        ]
        for masses in all_masses[1:]:
            idx = search_sorted_closest(masses, isotope_data["mass"])
            _isotopes = isotope_data[np.abs(masses[idx] - isotope_data["mass"]) < 0.05]
            isotopes = np.intersect1d(isotopes, _isotopes)

        self.table.setEnabledIsotopes(isotopes)


class NuVitesseImportWizard(QtWidgets.QWizard):
    page_paths = 0
    page_isotopes = 1

    laserImported = QtCore.Signal(Path, tuple)

    def __init__(self, paths: list[Path], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        if self.wizardStyle() != QtWidgets.QWizard.WizardStyle.MacStyle:
            self.setWizardStyle(QtWidgets.QWizard.WizardStyle.ModernStyle)
        self.setWindowTitle("Nu Vitesse Import")
        self.setMinimumSize(860, 680)

        self.setPage(self.page_paths, LaserImagePathsPage(paths, parent=self))
        self.setPage(self.page_isotopes, IsotopeSelectionPage(parent=self))

    def accept(self):
        paths = self.field("paths")
        datas = self.field("laserdata")
        all_masses = self.field("masses")
        infos = self.field("laserinfo")
        positions = self.field("positions")

        isotopes = self.field("selectedIsotopes")

        for data, masses, laser_info, pos, path in zip(
            datas, all_masses, infos, positions, paths
        ):
            idx = search_sorted_closest(masses, isotopes["mass"])

            dtype = [(f"{iso['isotope']}{iso['symbol']}", float) for iso in isotopes]
            data = rfn.unstructured_to_structured(data[..., idx], dtype=dtype)

            first_line = laser_info["LaserLineInfo"][0]

            if first_line["lt"] >= 2:  # vertical
                spotsize_x = np.median(
                    np.diff([li["sx"] for li in laser_info["LaserLineInfo"]])
                )
                spotsize_y = first_line["ss"]
            else:
                spotsize_x = first_line["ss"]
                spotsize_y = np.median(
                    np.diff([li["sy"] for li in laser_info["LaserLineInfo"]])
                )

            if first_line["imna"] != "":
                name = first_line["imna"]
            else:
                name = path.stem

            info = {
                "Name": name,
                "File Path": str(path.resolve()),
                "Import Date": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())
                ),
                "Import Path": str(path.resolve()),
                "Import Version pewlib": version("pewlib"),
                "Import Version pew2": version("pewpew"),
            }
            if "Metadata" in first_line:
                info.update({k: str(v) for k, v in first_line["Metadata"].items()})

            laser = Laser(data, config=SpotConfig(spotsize_x, spotsize_y), info=info)

            self.laserImported.emit(path, (laser, pos))

        super().accept()
