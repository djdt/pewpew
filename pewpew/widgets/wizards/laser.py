import datetime
import logging
from pathlib import Path

import numpy as np
from pewlib.config import SpotConfig
from pewlib.io.laser import read_iolite_laser_log, sync_data_with_laser_log
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
        self.setAcceptDrops(True)

        self._log_data: np.ndarray = np.array([])

        overview = (
            "This wizard will guide you through importing and aligning LA-ICP-MS data "
            "with an Iolite laser log, a file that records the laser line locations. "
            "To begin, select the path to the laser log file below."
        )

        label = QtWidgets.QLabel(overview)
        label.setWordWrap(True)

        self.path = PathSelectWidget(path, "LaserLog", [".csv"], "File")
        self.path.pathChanged.connect(self.completeChanged)

        self.radio_activeview = QtWidgets.QRadioButton("ActiveView2")
        self.radio_activeview.setChecked(True)
        self.radio_chromium = QtWidgets.QRadioButton("Chromium2")

        gbox_style = QtWidgets.QGroupBox("Log Style")
        gbox_style_layout = QtWidgets.QVBoxLayout()
        gbox_style_layout.addWidget(self.radio_activeview)
        gbox_style_layout.addWidget(self.radio_chromium)
        gbox_style.setLayout(gbox_style_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.path)
        layout.addWidget(gbox_style)
        layout.addStretch(1)
        self.setLayout(layout)

        self.registerField("laserlog", self, "log_prop")
        self.registerField("styleActiveView2", self.radio_activeview)
        self.registerField("styleChromium2", self.radio_chromium)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.suffix.lower() == ".csv":
                    event.accept()
                    return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.suffix.lower() == ".csv":
                    event.accept()
                    self.path.addPath(path)
                    return
        super().dropEvent(event)

    def isComplete(self) -> bool:
        return self.path.isComplete()

    def getLog(self) -> np.ndarray:
        return self._log_data

    def setLog(self, log_data: np.ndarray) -> None:
        self._log_data = log_data
        self.logChanged.emit()

    def validatePage(self) -> bool:
        log_style = "activeview2" if self.radio_activeview.isChecked() else "chromium2"
        log_data = read_iolite_laser_log(self.path.path, log_style=log_style)
        self.setField("laserlog", log_data)

        return True

    log_prop = QtCore.Property("QVariant", getLog, setLog, notify=logChanged)  # type: ignore


class LaserGroupListItem(QtWidgets.QListWidgetItem):
    def __init__(self, seq: int, comment: str, num: int) -> None:
        super().__init__()

        self.seq = seq

        self.lasers = QtWidgets.QListWidget()

        label = QtWidgets.QLabel(f"Sequence {seq} :: {comment}\nlines = {num}")

        layout = QtWidgets.QHBoxLayout()

        gbox = QtWidgets.QGroupBox("Lasers")
        gbox_layout = QtWidgets.QVBoxLayout()
        gbox_layout.addWidget(self.lasers)
        gbox.setLayout(gbox_layout)

        layout.addWidget(label, 0)
        layout.addWidget(gbox, 1)
        self.setLayout(layout)


class LaserGroupsImportPage(QtWidgets.QWizardPage):
    groupsChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self._datas: list[np.ndarray] = []
        self._infos: list[dict] = []

        self.checkbox_split = QtWidgets.QCheckBox("Split data into rows.")
        self.checkbox_split.clicked.connect(self.initializePage)

        self.group_tree = QtWidgets.QTreeWidget()
        self.group_tree.setColumnCount(3)
        self.group_tree.setHeaderLabels(["Sequence", "Name", "No. Lines"])
        self.group_tree.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.group_tree.setDragEnabled(True)
        self.group_tree.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        self.group_tree.model().dataChanged.connect(self.groupsChanged)

        root = self.group_tree.invisibleRootItem()
        root.setFlags(root.flags() ^ QtCore.Qt.ItemFlag.ItemIsDropEnabled)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.group_tree)
        layout.addWidget(self.checkbox_split)
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
                log_data[(log_data["sequence"] == i) & (log_data["state"] == 1)]
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
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsDropEnabled
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            self.group_tree.addTopLevelItem(item)

        datas = self.field("laserdata")
        infos = self.field("laserinfo")

        order = np.arange(len(datas))

        valid_date_fields = ["Acquisition Date"]  # only Agilent so far
        for field in valid_date_fields:
            if all(field in info for info in infos):
                info_times = np.array(
                    [
                        datetime.datetime.fromisoformat(info[field]).replace(
                            tzinfo=None  # drop timezone, no such thing in laser
                        )
                        for info in infos
                    ],
                    dtype=np.datetime64,
                )
                order = np.argsort(info_times)

        tree_idx = 0
        for idx in order:
            info = infos[idx]
            data = datas[idx]
            for row in range(data.shape[0] if self.checkbox_split.isChecked() else 1):
                item = self.group_tree.topLevelItem(
                    tree_idx % self.group_tree.topLevelItemCount()
                )
                child = QtWidgets.QTreeWidgetItem()
                child.setText(0, "---")
                child.setIcon(1, QtGui.QIcon.fromTheme("handle-sort"))
                child.setText(1, info["Name"])
                child.setData(1, QtCore.Qt.ItemDataRole.UserRole, idx)
                if self.checkbox_split.isChecked():
                    child.setText(2, f"row {row + 1}")
                    child.setData(2, QtCore.Qt.ItemDataRole.UserRole, row)
                else:
                    child.setData(2, QtCore.Qt.ItemDataRole.UserRole, -1)
                child.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsSelectable
                    | QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsDragEnabled
                )
                if item is None:
                    raise ValueError("missing item")
                item.addChild(child)
                tree_idx += 1

        item = QtWidgets.QTreeWidgetItem()
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, -1)
        item.setText(0, "None")
        item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled
        )
        self.group_tree.addTopLevelItem(item)

        self.group_tree.expandAll()

    def validatePage(self) -> bool:
        print("validate start")
        groups = self.field("groups")

        log = self.field("laserlog")

        params = self.field("laserparam")

        text_color = self.palette().color(
            QtGui.QPalette.ColorGroup.Normal, QtGui.QPalette.ColorRole.Text
        )
        for i in range(self.group_tree.topLevelItemCount()):
            item = self.group_tree.topLevelItem(i)
            if item is not None:
                item.setForeground(1, QtGui.QBrush(text_color))

        # Find the maximum time recorded in data
        invalid_items = {}
        for seq, idx in groups.items():
            log_max_time = np.ptp(
                log[np.isin(log["sequence"], seq)]["time"].astype(float) / 1000.0
            )
            seq_times = [
                params[i]["times"][j] if j > -1 else params[i]["times"].flat
                for i, j in idx
            ]
            seq_max_time = np.amax(seq_times)

            # Check if this time is less than requested by the log
            items = self.group_tree.findItems(
                str(seq), QtCore.Qt.MatchFlag.MatchExactly, 0
            )
            for item in items:
                if log_max_time > seq_max_time:
                    item.setForeground(1, QtGui.QBrush(QtCore.Qt.GlobalColor.red))
                    invalid_items[item.data(1, QtCore.Qt.ItemDataRole.DisplayRole)] = (
                        seq_max_time,
                        log_max_time,
                    )
                else:
                    item.setForeground(1, QtGui.QBrush(text_color))

        print("validate complete")
        if len(invalid_items) > 0:
            text = "\n".join(
                f"{k}: {v[0]:.2f} s > {v[1]:.2f} s" for k, v in invalid_items.items()
            )
            response = QtWidgets.QMessageBox.warning(
                self,
                "Invalid Laser Data Match",
                "Some laser patterns have longer acquistion times than their assigned ICP data files.\n"
                + text,
                buttons=QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if response == QtWidgets.QMessageBox.StandardButton.Cancel:
                return False
        return True

    def getGroups(self) -> dict[int, list[tuple[int, int]]]:
        """Returns dict of sequence: idx of 'laserdata'"""
        groups: dict[int, list[tuple[int, int]]] = {}
        for i in range(self.group_tree.topLevelItemCount()):
            item = self.group_tree.topLevelItem(i)
            if item is None:
                raise ValueError("missing item")
            if item.checkState(0) == QtCore.Qt.CheckState.Checked:
                seq = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                idx = [
                    (
                        item.child(j).data(1, QtCore.Qt.ItemDataRole.UserRole),
                        item.child(j).data(2, QtCore.Qt.ItemDataRole.UserRole),
                    )
                    for j in range(item.childCount())
                ]
                if len(idx) > 0:
                    groups[seq] = idx
        return groups

    groups_prop = QtCore.Property("QVariant", getGroups, notify=groupsChanged)  # type: ignore


class LaserLogImagePage(QtWidgets.QWizardPage):
    laserItemsChanged = QtCore.Signal()

    def __init__(
        self,
        options: GraphicsOptions | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        if options is None:
            options = GraphicsOptions()

        self.graphics = LaserGraphicsView(options, parent=self)
        self.graphics.setMinimumSize(QtCore.QSize(640, 480))

        self.spinbox_delay = QtWidgets.QDoubleSpinBox()
        self.spinbox_delay.setMinimum(-1.0)
        self.spinbox_delay.setMaximum(100.0)
        self.spinbox_delay.setDecimals(2)
        self.spinbox_delay.setSingleStep(1.0)
        self.spinbox_delay.setSuffix(" ms")
        self.spinbox_delay.setSpecialValueText("Auto")
        self.spinbox_delay.setMinimumWidth(
            QtGui.QFontMetrics(self.spinbox_delay.font())
            .boundingRect("Automatic (00.00)")
            .width()
            + self.spinbox_delay.baseSize().width()
        )
        self.spinbox_delay.valueChanged.connect(self.initializePage)

        self.spinbox_correction = QtWidgets.QDoubleSpinBox()
        self.spinbox_correction.setMinimum(-100.0)
        self.spinbox_correction.setMaximum(100.0)
        self.spinbox_correction.setDecimals(2)
        self.spinbox_correction.setSuffix(" ms")
        self.spinbox_correction.setSingleStep(1.0)
        self.spinbox_correction.valueChanged.connect(self.initializePage)

        self.checkbox_collapse = QtWidgets.QCheckBox("Remove space between images.")
        self.checkbox_collapse.checkStateChanged.connect(self.initializePage)
        self.checkbox_collapse.checkStateChanged.connect(self.graphics.zoomReset)

        controls_box = QtWidgets.QGroupBox("Import Options")
        controls_box_layout = QtWidgets.QFormLayout()
        controls_box_layout.addRow("Delay", self.spinbox_delay)
        controls_box_layout.addRow("Drift", self.spinbox_correction)
        controls_box_layout.addRow(self.checkbox_collapse)
        controls_box.setLayout(controls_box_layout)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(controls_box, 0)
        layout.addWidget(self.graphics, 1)
        self.setLayout(layout)

        self.registerField("laseritems", self, "laser_item_prop")

    def initializePage(self) -> None:
        log = self.field("laserlog")
        datas = self.field("laserdata")
        params = self.field("laserparam")
        infos = self.field("laserinfo")
        groups = self.field("groups")

        for item in self.graphics.laserItems():
            item.close()

        if self.spinbox_delay.value() < 0.0:
            delay = None
        else:
            delay = self.spinbox_delay.value() / 1000.0

        corr = self.spinbox_correction.value() / 1000.0

        extents = QtCore.QRectF()
        for seq, idx in groups.items():
            seq_datas = []
            seq_times = []
            for i, r in idx:
                x = datas[i]
                t = params[i]["times"]

                if r == -1:
                    seq_datas.append(x.flat)
                    seq_times.append(t.flat + np.linspace(0.0, corr, t.size))
                else:
                    seq_datas.append(x[r])
                    seq_times.append(t[r] + np.linspace(0.0, corr, t.shape[1]))
            data = np.concatenate(seq_datas)
            times = np.concatenate(seq_times)

            sync, sync_params = sync_data_with_laser_log(
                data, times, log, delay=delay, sequence=seq
            )
            if delay is None:
                self.spinbox_delay.setValue(-1.0)
                self.spinbox_delay.setSpecialValueText(
                    f"Auto ({sync_params['delay'] * 1000.0:.2f})"
                )

            laser = Laser(
                sync, info=infos[idx[0][0]], config=SpotConfig(*sync_params["spotsize"])
            )
            laser_item = LaserImageItem(laser, self.graphics.options)
            laser_item.setFlag(
                QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False
            )
            laser_item.setFlag(
                QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False
            )
            laser_item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            laser_item.setPos(*sync_params["origin"])
            if self.checkbox_collapse.isChecked():
                if extents.isNull():  # move to first image pos
                    extents.moveTo(*sync_params["origin"])
                rect = laser_item.sceneBoundingRect()
                if rect.left() >= extents.right():
                    rect.moveLeft(extents.right())
                elif rect.right() <= extents.left():
                    rect.moveRight(extents.left())
                if rect.top() >= extents.bottom():
                    rect.moveTop(extents.bottom())
                elif rect.bottom() <= extents.top():
                    rect.moveBottom(extents.top())
                extents = extents.united(rect)
                laser_item.setPos(rect.topLeft())
            laser_item.redraw()
            laser_item.setEnabled(False)
            self.graphics.scene().addItem(laser_item)

        self.laserItemsChanged.emit()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.graphics.zoomReset()

    def getLaserItems(self) -> list[LaserImageItem]:
        return self.graphics.laserItems()

    laser_item_prop = QtCore.Property(
        "QVariant", getLaserItems, notify=laserItemsChanged
    )


class LaserLogImportWizard(QtWidgets.QWizard):
    page_laser = 0
    page_format = 1
    page_agilent = 2
    page_csv = 3
    page_numpy = 4
    page_perkinelmer = 5
    page_text = 6
    page_thermo = 7
    page_nu = 8
    page_groups = 9
    page_image = 10

    laserImported = QtCore.Signal(Path, tuple)

    def __init__(
        self,
        path: Path | str = "",
        laser_paths: list[Path | str] | None = None,
        config: SpotConfig | None = None,
        options: GraphicsOptions | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Laser Log Import")
        self.setMinimumSize(860, 680)

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
                # "csv": self.page_csv,
                # "numpy": self.page_numpy,
                "nu": self.page_nu,
                "perkinelmer": self.page_perkinelmer,
                # "text": self.page_text,
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
        # self.setPage(
        #     self.page_csv,
        #     PathAndOptionsPage(
        #         paths, "csv", nextid=self.page_groups, multiplepaths=True, parent=self
        #     ),
        # )
        self.setPage(
            self.page_numpy,
            PathAndOptionsPage(
                paths, "numpy", nextid=self.page_groups, multiplepaths=True, parent=self
            ),
        )
        self.setPage(
            self.page_nu,
            PathAndOptionsPage(
                paths, "nu", nextid=self.page_groups, multiplepaths=True, parent=self
            ),
        )
        # self.setPage(
        #     self.page_perkinelmer,
        #     PathAndOptionsPage(
        #         paths,
        #         "perkinelmer",
        #         nextid=self.page_groups,
        #         multiplepaths=True,
        #         parent=self,
        #     ),
        # )
        # self.setPage(
        #     self.page_text,
        #     PathAndOptionsPage(
        #         paths, "text", nextid=self.page_groups, multiplepaths=True, parent=self
        #     ),
        # )
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

    def accept(self) -> None:
        items: list[LaserImageItem] = self.field("laseritems")
        for item in items:
            self.laserImported.emit(
                item.laser.info["File Path"], (item.laser, item.pos())
            )

        super().accept()
