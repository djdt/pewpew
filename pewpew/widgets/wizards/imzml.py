import logging
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
from pewlib.config import SpotConfig
from pewlib.io.imzml import ImzML, fast_parse_imzml
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction
from pewpew.charts.spectra import SpectraView
from pewpew.graphics.colortable import get_table
from pewpew.graphics.imageitems import ScaledImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions
from pewpew.lib.numpyqt import NumpyRecArrayTableModel
from pewpew.validators import DoublePrecisionDelegate, DoubleValidatorWithEmpty
from pewpew.widgets.wizards.options import PathSelectWidget

logger = logging.getLogger(__name__)


class ClickableImageItem(ScaledImageItem):
    clickedAtPosition = QtCore.Signal(QtCore.QPoint)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapToData(event.pos())
            self.clickedAtPosition.emit(pos)
            event.accept()
        else:
            super().mousePressEvent(event)


class MassTable(QtWidgets.QTableView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        array = np.array([np.nan], dtype=[("m/z", float)])
        self.setModel(NumpyRecArrayTableModel(array))
        self.setItemDelegate(
            DoublePrecisionDelegate(4, validator=DoubleValidatorWithEmpty())
        )
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        self.model().dataChanged.connect(self.insertOrDeleteLastRows)

        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)

    def insertOrDeleteLastRows(self) -> None:
        array = self.model().array
        if not np.isnan(array["m/z"][-1]):
            self.model().insertRow(self.model().rowCount())
        else:
            size = self.model().rowCount()
            last_real = np.argmax(~np.isnan(array["m/z"][::-1]))
            if last_real != 0:
                last_real = size - last_real
            if last_real == size - 1:
                return
            self.model().removeRows(last_real, self.model().rowCount() - last_real - 1)

    def addMass(self, mass: float) -> None:
        index = self.model().index(self.model().rowCount() - 1, 0)
        self.model().setData(index, mass, QtCore.Qt.ItemDataRole.EditRole)

    def targetMasses(self) -> np.ndarray:
        masses = self.model().array["m/z"]
        return masses[~np.isnan(masses)]

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            text = QtWidgets.QApplication.clipboard().text("plain")[0]
            selection = self.selectedIndexes()
            start_row = min(selection, key=lambda i: i.row()).row()

            for row, row_text in enumerate(text.split("\n")):
                val = row_text.split("\t", maxsplit=1)[0]
                if not self.model().hasIndex(start_row + row, 0):
                    self.model().insertRow(start_row + row)
                self.model().setData(
                    self.model().index(start_row + row, 0),
                    val,
                    QtCore.Qt.ItemDataRole.EditRole,
                )
        elif event.matches(QtGui.QKeySequence.StandardKey.Delete):
            selection = self.selectedIndexes()
            indicies = [QtCore.QPersistentModelIndex(i) for i in selection]
            for index in indicies:
                self.model().removeRow(index.row())
            self.clearSelection()
        else:
            super().keyPressEvent(event)


class ImzMLImportPage(QtWidgets.QWizardPage):
    imzmlChanged = QtCore.Signal()

    def __init__(
        self,
        imzml: Path,
        external_binary: Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setTitle("Import Laser File")
        self.setAcceptDrops(True)

        self._log_data: np.ndarray = np.array([])

        overview = (
            "This wizard will guide you through importing MSI data stoed in an imzML. "
            "To begin, select the path to the .imzML and binary (.ibd) file below."
        )

        label = QtWidgets.QLabel(overview)
        label.setWordWrap(True)

        self.path = PathSelectWidget(imzml, "imzML", [".imzML"], "File")
        self.path.pathChanged.connect(self.guessBinaryPath)
        self.path.pathChanged.connect(self.completeChanged)

        if external_binary is None and imzml.is_file() and imzml.exists():
            external_binary = imzml.with_suffix(".ibd")
        if external_binary is None:
            external_binary = Path("")
        self.path_binary = PathSelectWidget(
            external_binary, "Binary Data", [".ibd"], "File"
        )
        self.path_binary.pathChanged.connect(self.completeChanged)

        self._imzml: ImzML = None

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.path)
        layout.addWidget(self.path_binary)
        layout.addStretch(1)
        self.setLayout(layout)

        self.registerField("imzml_path", self.path.lineedit_path)
        self.registerField("imzml", self, "imzml_prop")

    def isComplete(self) -> bool:
        return self.path.isComplete() and self.path_binary.isComplete()

    def guessBinaryPath(self) -> None:
        if (
            self.path.path.name != ""
            and self.path.path.with_suffix(".ibd").exists()
            and self.path_binary.path == Path()
        ):
            self.path_binary.addPath(self.path.path.with_suffix(".ibd"))

    def validatePage(self) -> bool:
        file_size = self.path.path.stat().st_size
        dlg = QtWidgets.QProgressDialog("Parsing imzML", "Cancel", 0, file_size)
        dlg.setWindowTitle("imzML Import")
        dlg.setMinimumWidth(320)
        dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        def update_progress(tell: int) -> bool:
            dlg.setValue(tell)
            return not dlg.wasCanceled()

        try:
            imzml = fast_parse_imzml(
                self.path.path, self.path_binary.path, callback=update_progress
            )
        except UserWarning:
            return False

        self.setField("imzml", imzml)
        dlg.close()
        return True

    def getImzML(self) -> ImzML:
        return self._imzml

    def setImzML(self, imzml: ImzML) -> None:
        self._imzml = imzml

    imzml_prop = QtCore.Property("QVariant", getImzML, setImzML, notify=imzmlChanged)


class ImzMLTargetMassPage(QtWidgets.QWizardPage):
    targetMassesChanged = QtCore.Signal()

    def __init__(
        self,
        options: GraphicsOptions | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        if options is None:
            options = GraphicsOptions()

        self.image: ClickableImageItem | None = None
        self.image_cache: dict[float, np.ndarray] = {}

        self.mass_table = MassTable()
        self.mass_table.model().dataChanged.connect(self.completeChanged)
        self.mass_table.model().rowsRemoved.connect(self.completeChanged)
        self.mass_table.clicked.connect(self.massSelected)

        self.mass_width = QtWidgets.QSpinBox()
        self.mass_width.setRange(0, 1000)
        self.mass_width.setValue(10)
        self.mass_width.setSingleStep(10)
        self.mass_width.setSuffix(" ppm")
        self.mass_width.valueChanged.connect(self.completeChanged)
        self.mass_width.valueChanged.connect(self.image_cache.clear)

        self.graphics = LaserGraphicsView(options, parent=self)
        self.graphics.setMinimumSize(QtCore.QSize(640, 320))

        self.spectra = SpectraView()

        self.action_tic = qAction(
            "black_sum", "Draw TIC", "Draw the total-ion-chromatogram.", self.drawTIC
        )
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.addAction(self.action_tic)
        self.toolbar.setParent(self.graphics)
        self.toolbar.widgetForAction(self.action_tic).setAutoFillBackground(True)

        self.registerField("mass_width", self.mass_width)
        self.registerField("target_masses", self, "target_masses_prop")

        layout_mass_width = QtWidgets.QHBoxLayout()
        layout_mass_width.addWidget(QtWidgets.QLabel("Mass width:"), 0)
        layout_mass_width.addWidget(self.mass_width, 1)

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addWidget(QtWidgets.QLabel("Target masses"), 0)
        layout_left.addWidget(self.mass_table, 1)
        layout_left.addLayout(layout_mass_width, 0)

        layout_right = QtWidgets.QVBoxLayout()
        # layout_right.addWidget(self.toolbar)
        layout_right.addWidget(self.graphics, 1)
        layout_right.addWidget(self.spectra, 0)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_left, 0)
        layout.addLayout(layout_right, 1)

        self.setLayout(layout)

    def initializePage(self) -> None:
        self.drawTIC()

    def isComplete(self) -> bool:
        return (
            self.mass_width.hasAcceptableInput()
            and len(self.mass_table.targetMasses()) > 0
        )

    def getTargetMasses(self) -> np.ndarray:
        return self.mass_table.targetMasses()

    def drawImage(self, image: np.ndarray | ClickableImageItem) -> None:
        if self.image is not None:
            self.graphics.scene().removeItem(self.image)

        if isinstance(image, np.ndarray):
            imzml: ImzML = self.field("imzml")
            sx, sy = imzml.scan_settings.image_size
            px, py = imzml.scan_settings.pixel_size
            rect = QtCore.QRectF(0, 0, sx * px, sy * py)
            image = ClickableImageItem.fromArray(
                image, rect, list(get_table(self.graphics.options.colortable))
            )

        self.image = image
        self.image.clickedAtPosition.connect(self.drawSpectraAtPos)
        self.image.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False
        )
        self.image.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False
        )
        self.image.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False
        )
        self.graphics.scene().addItem(self.image)
        self.graphics.zoomReset()

    def drawTIC(self) -> None:
        imzml: ImzML = self.field("imzml")
        tic = imzml.extract_tic()
        tic -= np.nanmin(tic)
        tic /= np.nanmax(tic)

        self.drawImage(tic)

    def drawMass(self, mz: float) -> None:
        if mz not in self.image_cache:
            imzml: ImzML = self.field("imzml")
            img = imzml.extract_masses(
                mz, mass_width_ppm=float(self.mass_width.value())
            )[:, :, 0]
            self.image_cache[mz] = img
        else:
            img = self.image_cache[mz]

        img -= np.nanmin(img)
        img /= np.nanmax(img)
        self.drawImage(img)

    def drawSpectraAtPos(self, pos: QtCore.QPoint) -> None:
        self.spectra.clear()

        imzml: ImzML = self.field("imzml")

        # convert mapToData pos to stored spectrum pos
        px, py = pos.x() + 1, pos.y() + 1
        try:
            spectrum = imzml.spectra[(px, py)]
        except KeyError:
            return

        x = spectrum.get_binary_data(
            imzml.mz_params.id, imzml.mz_params.dtype, imzml.external_binary
        )
        y = spectrum.get_binary_data(
            imzml.intensity_params.id,
            imzml.intensity_params.dtype,
            imzml.external_binary,
        )
        spec = self.spectra.drawCentroidSpectra(x, y)
        spec.mzClicked.connect(self.mass_table.addMass)
        spec.mzDoubleClicked.connect(self.drawMass)

    target_masses_prop = QtCore.Property(
        "QVariant", getTargetMasses, notify=targetMassesChanged
    )

    def massSelected(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return
        text = self.mass_table.model().data(index, QtCore.Qt.ItemDataRole.EditRole)
        if text == "":
            return
        self.drawMass(float(text))


class ImzMLImportWizard(QtWidgets.QWizard):
    page_imzml = 0
    page_masses = 1

    laserImported = QtCore.Signal(Path, Laser)

    def __init__(
        self,
        path: Path | str = "",
        binary_path: Path | str | None = None,
        config: SpotConfig | None = None,
        options: GraphicsOptions | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("ImzML Import")
        self.setMinimumSize(860, 680)

        if isinstance(path, str):
            path = Path(path)
        if isinstance(binary_path, str):
            binary_path = Path(binary_path)

        self.setPage(self.page_imzml, ImzMLImportPage(path, binary_path))
        self.setPage(self.page_masses, ImzMLTargetMassPage(options))

    def accept(self) -> None:
        path = Path(self.field("imzml_path"))
        imzml: ImzML = self.field("imzml")
        mass_width = float(self.field("mass_width"))
        target_masses: np.ndarray = self.field("target_masses")

        # cleanup the masses
        mass_range = imzml.mass_range()
        target_masses = target_masses[
            (target_masses > mass_range[0]) & (target_masses < mass_range[1])
        ]
        if len(target_masses) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No masses to import!",
                "Check targets are within mass range, "
                f"{mass_range[0]:.4f} - {mass_range[1]:.4f}.",
            )
            return
        target_masses = np.unique(target_masses)

        data = imzml.extract_masses(target_masses, mass_width)

        data = rfn.unstructured_to_structured(
            data, names=[f"{x:.4f}" for x in target_masses]
        )

        laser = Laser(
            data,
            config=SpotConfig(
                imzml.scan_settings.pixel_size[0], imzml.scan_settings.pixel_size[1]
            ),
            info={
                "Name": path.stem,
                "File Path": str(path.resolve()),
                "Import Date": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())
                ),
                "Import Path": str(path.resolve()),
                "Import Version pewlib": version("pewlib"),
                "Import Version pew2": version("pewpew"),
            },
        )
        self.laserImported.emit(laser.info["File Path"], laser)

        super().accept()
