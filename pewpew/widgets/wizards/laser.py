import logging
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
from pewlib.config import SpotConfig
from pewlib.io.laser import read_nwi_laser_log
from pewlib.laser import Laser
from pewlib.process import peakfinding
from pewlib.process.calc import view_as_blocks
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.charts.colors import sequential
from pewpew.charts.signal import SignalChart
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions
from pewpew.validators import DecimalValidator, DecimalValidatorNoZero, OddIntValidator
from pewpew.widgets.dialogs import NameEditDialog
from pewpew.widgets.wizards.import_ import FormatPage
from pewpew.widgets.wizards.options import PathAndOptionsPage, PathSelectWidget

logger = logging.getLogger(__name__)


class LaserLogImportPage(QtWidgets.QWizardPage):
    logChanged = QtCore.Signal()

    def __init__(self, path: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setTitle("Import Laser File")

        self._log_data: np.ndarray = np.array([])

        overview = (
            "This wizard will guide you through importing and aligning LA-ICP-MS data "
            "with a laser log, a file that records the laser line locations. "
            "To begin, select the path to the laser log file below."
        )

        label = QtWidgets.QLabel(overview)
        label.setWordWrap(True)

        self.path = PathSelectWidget(path, "LaserLog", [".csv"], "File")
        self.path.pathChanged.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.path)
        self.setLayout(layout)

        self.registerField("laserlog", self, "log_prop")

    def isComplete(self) -> bool:
        return self.path.isComplete()

    def getLog(self) -> np.ndarray:
        return self._log_data

    def setLog(self, log_data: np.ndarray) -> None:
        self._log_data = log_data
        self.logChanged.emit()

    def validatePage(self) -> bool:
        log_data = read_nwi_laser_log(self.path.path)
        self.setField("laserlog", log_data)

        return True

    log_prop = QtCore.Property("QVariant", getLog, setLog, notify=logChanged)


# class LaserGroupTreeItem(QtWidgets.QTreeWidgetItem):
#     def __init__(self, seq: int, comment: str, num: int) -> None:
#         super().__init__(QtWidgets.QTreeWidgetItem.ItemType.UserType)
#
#         self.seq = seq
#
#         label = QtWidgets.QLabel(f"Sequence {seq} :: {comment}")
#         num_label = QtWidgets.QLabel(f"Num lines = {num}")
#
#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(label)
#         layout.addWidget(num_label)
#         self.setLayout(layout)
#


class LaserGroupsImportPage(QtWidgets.QWizardPage):
    dataChanged = QtCore.Signal()
    infoChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self._datas = []
        self._infos = []

        self.group_tree = QtWidgets.QTreeWidget()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.group_tree)
        self.setLayout(layout)

        self.registerField("laserdata", self, "data_prop")
        self.registerField("laserinfo", self, "info_prop")

    def initializePage(self) -> None:
        log_data = self.field("laserlog")
        sequences, seq_idx = np.unique(
            log_data["sequence"][log_data["sequence"] > 0], return_index=True
        )
        comments = log_data["comment"][seq_idx]
        num_lines = [
            np.count_nonzero(log_data[log_data["sequence"] == i]["state"] == "On")
            for i in seq_idx
        ]

        for seq, comment, num in zip(sequences, comments, num_lines):
            print(seq, comment, num)
            item = QtWidgets.QTreeWidgetItem(f"Sequence {seq} :: {comment} :: {num} lines.")
            self.group_tree.addTopLevelItem(item)

        # self.

    def getData(self) -> list[np.ndarray]:
        if len(self._datas) == 0:
            return [np.array([], dtype=[("", np.float64)])]
        return np.concatenate([d.ravel() for d in self._datas], axis=0)

    def setData(self, datas: list[np.ndarray]) -> None:
        self._datas = datas
        self.dataChanged.emit()

    def getInfo(self) -> list[dict]:
        return self._infos

    def setInfo(self, infos: list[dict]) -> None:
        self._infos = infos
        self.infoChanged.emit()

    data_prop = QtCore.Property("QVariant", getData, setData, notify=dataChanged)
    info_prop = QtCore.Property("QVariant", getInfo, setInfo, notify=infoChanged)


class LaserLogImportWizard(QtWidgets.QWizard):
    page_laser = 0
    page_format = 1
    page_agilent = 2
    page_csv = 3
    page_numpy = 4
    page_perkinelmer = 5
    page_text = 6
    page_thermo = 7
    page_groups = 8
    page_spot_image = 9
    page_spot_config = 10

    laserImported = QtCore.Signal(Laser)

    def __init__(
        self,
        path: Path | str = "",
        laser_paths: list[Path | str] | None = None,
        config: SpotConfig | None = None,
        options: GraphicsOptions | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        if isinstance(path, str):
            path = Path(path)
        if laser_paths is None:
            laser_paths = []
        paths = [Path(x) for x in laser_paths]

        self.setPage(self.page_laser, LaserLogImportPage(path))

        format_page = FormatPage(
            "Select format of laser file(s) for import.",
            page_id_dict={
                "agilent": self.page_agilent,
                "csv": self.page_csv,
                "numpy": self.page_numpy,
                "perkinelmer": self.page_perkinelmer,
                "text": self.page_text,
                "thermo": self.page_thermo,
            },
            parent=self,
        )

        self.setPage(self.page_format, format_page)
        self.setPage(
            self.page_agilent,
            PathAndOptionsPage(
                paths,
                "agilent",
                nextid=self.page_groups,
                multiplepaths=True,
                parent=self,
            ),
        )
        self.setPage(
            self.page_csv,
            PathAndOptionsPage(
                paths, "csv", nextid=self.page_groups, multiplepaths=True, parent=self
            ),
        )
        self.setPage(
            self.page_numpy,
            PathAndOptionsPage(
                paths, "numpy", nextid=self.page_groups, multiplepaths=True, parent=self
            ),
        )
        self.setPage(
            self.page_perkinelmer,
            PathAndOptionsPage(
                paths,
                "perkinelmer",
                nextid=self.page_groups,
                multiplepaths=True,
                parent=self,
            ),
        )
        self.setPage(
            self.page_text,
            PathAndOptionsPage(
                paths, "text", nextid=self.page_groups, multiplepaths=True, parent=self
            ),
        )
        self.setPage(
            self.page_thermo,
            PathAndOptionsPage(
                paths,
                "thermo",
                nextid=self.page_groups,
                multiplepaths=True,
                parent=self,
            ),
        )
        self.setPage(self.page_groups, LaserGroupsImportPage())
