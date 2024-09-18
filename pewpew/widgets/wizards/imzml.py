import logging
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from pewlib.config import SpotConfig
from pewlib.io.imzml import MZML_NS, ImzML, ParamGroup, ScanSettings, Spectrum
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets

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
        # todo delegate with delete
        self.setItemDelegate(
            DoublePrecisionDelegate(4, validator=DoubleValidatorWithEmpty())
        )

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
            "This wizard will guide you through importing MSI data stoed in an imzML."
            "To begin, select the path to the .imzML and binary (.ibd) file below."
        )

        label = QtWidgets.QLabel(overview)
        label.setWordWrap(True)

        self.path = PathSelectWidget(imzml, "imzML", [".imzML"], "File")
        self.path.pathChanged.connect(self.guessBinaryPath)
        self.path.pathChanged.connect(self.completeChanged)

        if external_binary is None:
            external_binary = imzml.with_suffix(".ibd")
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

        self.registerField("imzml", self, "imzml_prop")

    def isComplete(self) -> bool:
        return self.path.isComplete() and self.path_binary.isComplete()

    def guessBinaryPath(self) -> None:
        if (
            self.path.path.with_suffix(".ibd").exists()
            and self.path_binary.path == Path()
        ):
            self.path_binary.addPath(self.path.path.with_suffix(".ibd"))

    def validatePage(self) -> bool:
        file_size = self.path.path.stat().st_size
        dlg = QtWidgets.QProgressDialog("Parsing imzML", "Cancel", 0, file_size)
        dlg.setWindowTitle("imzML Import")
        dlg.setMinimumWidth(320)
        dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        spectra: dict[tuple[int, int], Spectrum] = {}

        fp = self.path.path.open()
        iter = ElementTree.iterparse(fp, events=("end",))
        for event, elem in iter:
            if dlg.wasCanceled():
                return False
            if elem.tag == f"{{{MZML_NS['mz']}}}scanSettingsList":
                scan = elem.find("mz:scanSettings", MZML_NS)
                scan_settings = ScanSettings.from_xml_element(scan)
                elem.clear()
            elif elem.tag == f"{{{MZML_NS['mz']}}}referenceableParamGroup":
                if (
                    elem.find(
                        # todo : fix
                        "mz:cvParam[@accession='MS:1000514']",
                        MZML_NS,
                        # f"mz:cvParam[@accession='{ParamGroup.mz_array_cv}']", MZML_NS
                    )
                    is not None
                ):
                    mz_params = ParamGroup.from_xml_element(elem)
                elif (
                    elem.find(
                        # todo : fix
                        # f"mz:cvParam[@accession='{ParamGroup.intensity_array_cv}']",
                        "mz:cvParam[@accession='MS:1000515']",
                        MZML_NS,
                    )
                    is not None
                ):
                    intensity_params = ParamGroup.from_xml_element(elem)
                elem.clear()
            elif elem.tag == f"{{{MZML_NS['mz']}}}spectrum":
                spectrum = Spectrum.from_xml_element(elem)
                spectra[(spectrum.x, spectrum.y)] = spectrum

                # Update dialog on spectrum ends
                dlg.setValue(fp.tell())
                elem.clear()

        dlg.close()

        imzml = ImzML(
            scan_settings,
            mz_params,
            intensity_params,
            spectra,
            external_binary=self.path_binary.path,
        )
        self.setField("imzml", imzml)
        return True

    def getImzML(self) -> ImzML:
        return self._imzml

    def setImzML(self, imzml: ImzML) -> None:
        self._imzml = imzml

    imzml_prop = QtCore.Property("QVariant", getImzML, setImzML, notify=imzmlChanged)


class ImzMLTargetMassPage(QtWidgets.QWizardPage):
    def __init__(
        self,
        options: GraphicsOptions | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        if options is None:
            options = GraphicsOptions()

        self.mass_table = MassTable()

        self.mass_width = QtWidgets.QSpinBox()
        self.mass_width.setRange(0, 1000)
        self.mass_width.setValue(10)
        self.mass_width.setSingleStep(10)
        self.mass_width.setSuffix(" ppm")

        self.graphics = LaserGraphicsView(options, parent=self)
        self.graphics.setMinimumSize(QtCore.QSize(640, 480))

        self.image: ClickableImageItem | None = None

        self.spectra = SpectraView()

        layout_mass_width = QtWidgets.QHBoxLayout()
        layout_mass_width.addWidget(QtWidgets.QLabel("Mass width:"), 0)
        layout_mass_width.addWidget(self.mass_width, 1)

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addWidget(QtWidgets.QLabel("Target masses"), 0)
        layout_left.addWidget(self.mass_table, 1)
        layout_left.addLayout(layout_mass_width, 0)

        layout_right = QtWidgets.QVBoxLayout()
        layout_right.addWidget(self.graphics, 1)
        layout_right.addWidget(self.spectra, 0)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_left, 0)
        layout.addLayout(layout_right, 1)

        self.setLayout(layout)

    def initializePage(self) -> None:
        self.drawTIC()

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
        imzml: ImzML = self.field("imzml")
        x = imzml.extract_masses(np.array([mz]))[:, :, 0]
        x -= np.nanmin(x)
        x /= np.nanmax(x)
        self.drawImage(x)

    def drawSpectraAtPos(self, pos: QtCore.QPoint) -> None:
        self.spectra.clear()

        imzml: ImzML = self.field("imzml")

        # convert mapToData pos to stored spectrum pos
        px, py = pos.x() + 1, imzml.scan_settings.image_size[1] - pos.y()
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


# app = QtWidgets.QApplication()
# wiz = QtWidgets.QWizard()
# wiz.addPage(
#     ImzMLImportPage(
#         Path("/home/tom/Downloads/slide 8 at 19%.imzML"),
#         Path("/home/tom/Downloads/slide 8 at 19%.ibd"),
#     )
# )
# wiz.addPage(ImzMLTargetMassPage())
# wiz.show()
# app.exec()
#
#
# app = QtWidgets.QApplication()
# w = SpectraView()
# w.drawCentroidSpectra(np.arange(100), np.random.random(100))
# w.spectra.mzClicked.connect(lambda x: print("click", x, flush=True))
# w.spectra.mzDoubleClicked.connect(lambda x: print("dbl click", x, flush=True))
# w.show()
# w = QtWidgets.QMainWindow()
# t = MassTable()
# w.setCentralWidget(t)
# app.exec()
