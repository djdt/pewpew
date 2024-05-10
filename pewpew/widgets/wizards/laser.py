import logging
from pathlib import Path

import numpy as np
from pewlib.config import SpotConfig
from pewlib.io.laser import read_nwi_laser_log
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.graphics.options import GraphicsOptions
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


class LaserGroupListItem(QtWidgets.QListWidgetItem):
    def __init__(self, seq: int, comment: str, num: int) -> None:
        super().__init__()

        self.seq = seq

        self.lasers = QtWidgets.QListWidget()

        label = QtWidgets.QLabel(f"Sequence {seq} :: {comment}\nlines = {num}")

        layout = QtWidgets.QHBoxLayout()

        gbox = QtWidgets.QGroupBox("Lasers")
        gbox.setLayout(QtWidgets.QVBoxLayout())
        gbox.layout().addWidget(self.lasers)
        layout.addWidget(label, 0)
        layout.addWidget(gbox, 1)
        self.setLayout(layout)


class LaserGroupsImportPage(QtWidgets.QWizardPage):
    dataChanged = QtCore.Signal()
    infoChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self._datas: list[np.ndarray] = []
        self._infos: list[dict] = []

        self.group_tree = QtWidgets.QTreeWidget()
        self.group_tree.setColumnCount(4)
        self.group_tree.setHeaderLabels(["Sequence", "Name", "No. Lines", "Lasers"])
        self.group_tree.setDragEnabled(True)
        self.group_tree.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )

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
            np.count_nonzero(
                log_data[(log_data["sequence"] == i) & (log_data["state"] == "On")]
            )
            for i in sequences
        ]

        self.group_tree.clear()
        for row, (seq, comment, num) in enumerate(zip(sequences, comments, num_lines)):
            item = QtWidgets.QTreeWidgetItem()
            item.setData(0, 0, seq)
            item.setText(0, str(seq))
            item.setText(1, comment)
            item.setText(2, str(num))
            item.setFlags(
                QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled
            )
            self.group_tree.addTopLevelItem(item)

        infos = self.field("laserinfo")
        for i, info in enumerate(infos):
            item = self.group_tree.topLevelItem(i % self.group_tree.topLevelItemCount())
            child = QtWidgets.QTreeWidgetItem()
            child.setData(0, 0, i)
            child.setText(0, "---")
            child.setText(1, "---")
            child.setText(2, "---")
            child.setIcon(3, QtGui.QIcon.fromTheme("drag-handle-symbolic"))
            child.setText(3, info["Name"])
            child.setFlags(
                QtCore.Qt.ItemIsSelectable
                | QtCore.Qt.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            item.addChild(child)

        self.group_tree.expandAll()

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

        # @nocommit
        path = Path("/home/tom/Downloads/LaserLog_24-05-02_13-49-34.csv")
        paths = [Path("/home/tom/Downloads/lasso.b/")]

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
