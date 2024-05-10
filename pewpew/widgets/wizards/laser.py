import logging
from pathlib import Path

import numpy as np
from pewlib.config import SpotConfig
from pewlib.io.laser import read_nwi_laser_log, sync_data_nwi_laser_log
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.graphics.imageitems import LaserImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
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
    groupsChanged = QtCore.Signal()

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
        self.group_tree.model().dataChanged.connect(self.groupsChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.group_tree)
        self.setLayout(layout)

        self.registerField("groups", self, "groups_prop")

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
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, seq)
            item.setText(0, str(seq))
            item.setCheckState(0, QtCore.Qt.CheckState.Checked)
            item.setText(1, comment)
            item.setText(2, str(num))
            item.setFlags(
                QtCore.Qt.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsDropEnabled
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            self.group_tree.addTopLevelItem(item)

        infos = self.field("laserinfo")
        for i, info in enumerate(infos):
            item = self.group_tree.topLevelItem(i % self.group_tree.topLevelItemCount())
            child = QtWidgets.QTreeWidgetItem()
            child.setData(0, QtCore.Qt.ItemDataRole.UserRole, i)
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

    def getGroups(self) -> dict[int, list[int]]:
        """Returns dict of sequence: idx of 'laserdata'"""
        groups: dict[int, list[int]] = {}
        for i in range(self.group_tree.topLevelItemCount()):
            item = self.group_tree.topLevelItem(i)
            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                seq = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                idx = [
                    item.child(j).data(0, QtCore.Qt.ItemDataRole.UserRole)
                    for j in range(item.childCount())
                ]
                if len(idx) > 0:
                    groups[seq] = idx
        return groups

    groups_prop = QtCore.Property("QVariant", getGroups, notify=groupsChanged)


class LaserLogImagePage(QtWidgets.QWizardPage):
    def __init__(
        self,
        options: GraphicsOptions | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        if options is None:
            options = GraphicsOptions()

        self.graphics = LaserGraphicsView(options, parent=self)
        self.graphics.show()

        self.spinbox_delay = QtWidgets.QDoubleSpinBox()
        self.spinbox_delay.setMinimum(0.0)
        self.spinbox_delay.setMaximum(10.0)
        self.spinbox_delay.setDecimals(4)
        self.spinbox_delay.setSingleStep(0.001)

        self.merge_sequences = QtWidgets.QCheckBox("Merge sequences.")
        self.merge_sequences.setChecked(True)

        controls_box = QtWidgets.QGroupBox("Import Options")
        controls_box.setLayout(QtWidgets.QFormLayout())
        controls_box.layout().addRow("Delay", self.spinbox_delay)
        controls_box.layout().addWidget(self.merge_sequences)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(controls_box, 0)
        layout.addWidget(self.graphics, 1)
        self.setLayout(layout)

    def initializePage(self) -> None:
        log = self.field("laserlog")
        datas = self.field("laserdata")
        params = self.field("laserparam")
        infos = self.field("laserinfo")
        groups = self.field("groups")

        for seq, idx in groups.items():
            data = np.concatenate([datas[i].flat for i in idx])
            if all("times" in params[i] for i in idx):
                times = np.concatenate([params[i]["times"] for i in idx])
            else:
                times = params[idx[0]]["scantime"]

            sync, params = sync_data_nwi_laser_log(data, times, log, sequence=seq)

            laser = Laser(
                sync, info=infos[idx[0]], config=SpotConfig(*params["spotsize"])
            )
            laser_item = LaserImageItem(laser, self.graphics.options)
            laser_item.setFlag(
                QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False
            )
            laser_item.setFlag(
                QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False
            )
            laser_item.redraw()
            laser_item.setPos(*params["origin"])
            self.graphics.scene().addItem(laser_item)
        self.graphics.zoomReset()


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
    page_image = 9

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
                register_laser_fields=True,
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
        self.setPage(self.page_image, LaserLogImagePage(options))
