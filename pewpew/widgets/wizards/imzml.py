import logging
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from pewlib.config import SpotConfig
from pewlib.io.imzml import ImzML
from pewlib.laser import Laser
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.charts.spectra import SpectraView
from pewpew.graphics.colortable import get_table
from pewpew.graphics.imageitems import ScaledImageItem
from pewpew.graphics.lasergraphicsview import LaserGraphicsView
from pewpew.graphics.options import GraphicsOptions
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


class ElementTreeParserThread(QtCore.QThread):
    importFinished = QtCore.Signal(ElementTree.ElementTree)
    importFailed = QtCore.Signal(str)
    progressChanged = QtCore.Signal(int)

    def __init__(self, path: Path | str, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.path = Path(path)

    def run(self) -> None:
        """Start the import thread."""
        iter = ElementTree.iterparse(self.path)

        for i, (event, elem) in enumerate(iter):
            if self.isInterruptionRequested():  # pragma: no cover
                break
            self.progressChanged.emit(i)

        self.importFinished.emit(elem.root)


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


    @classmethod
    def from_file_iterative(
        cls,
        path: Path | str,
        external_binary: Path | str,
        scan_number: int = 1,
        callback: None = None,
    ) -> "ImzML":
        iter = ElementTree.iterparse(path, events=("end",))

        spectra = {}

        for event, elem in iter:
            if elem.tag == f"{{{MZML_NS['mz']}}}scanSettingsList":
                scans = elem.findall("mz:scanSettings", MZML_NS)
                scan_settings = ScanSettings.from_xml_element(scans[scan_number - 1])
                elem.clear()
            elif elem.tag == f"{{{MZML_NS['mz']}}}referenceableParamGroup":
                if (
                    elem.find(
                        f"mz:cvParam[@accession='{CV_PARAMGROUP['MZ_ARRAY']}']", MZML_NS
                    )
                    is not None
                ):
                    mz_group = ParamGroup.from_xml_element(elem)
                elif (
                    elem.find(
                        f"mz:cvParam[@accession='{CV_PARAMGROUP['INTENSITY_ARRAY']}']",
                        MZML_NS,
                    )
                    is not None
                ):
                    intensity_group = ParamGroup.from_xml_element(elem)
                elem.clear()
            elif elem.tag == f"{{{MZML_NS['mz']}}}spectrum":
                spectrum = Spectrum.from_xml_element(elem)
                spectra[(spectrum.x, spectrum.y)] = spectrum
                elem.clear()
                if callback is not None:
                    if callback():
                        return
        return cls(
            scan_settings,
            mz_group,
            intensity_group,
            spectra,
            external_binary=external_binary,
        )

    def validatePage(self) -> bool:
        file_size = self.path.path.stat().st_size
        dlg = QtWidgets.QProgressDialog("Parsing imzML", "Cancel", 0, file_size)
        dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        """Callback."""
        fp = self.path.path.open()
        def callback():
            dlg.setValue(fp.tell())
            return dlg.wasCanceled()

        imzml = ImzML.from_file_iterative(fp, self.path_binary.path, callback=callback)
        # print("EXITING LOOP", flush=True)
        # dlg.setValue(dlg.maximum())
        # imzml = ImzML.from_etree(iter.root, self.path_binary.path)
        # ElementTree.parse(self.path.path)
        # imzml = ImzML.from_file(self.path.path, self.path_binary.path)
        t1= time.time()
        print("IMZML FIN", t1-t0, flush=True)
        self.setField("imzML", imzml)
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

        self.mass_list = QtWidgets.QListWidget()

        self.mass_width = QtWidgets.QLineEdit("10.0")
        self.mass_width.setValidator(QtGui.QDoubleValidator(0.0, 1000.0, 2))

        self.graphics = LaserGraphicsView(options, parent=self)
        self.graphics.setMinimumSize(QtCore.QSize(640, 480))
        self.graphics

        self.image: ClickableImageItem | None = None

        self.spectra = SpectraView()

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addWidget(self.mass_list, 1)
        layout_left.addWidget(self.mass_width, 0)

        layout_right = QtWidgets.QVBoxLayout()
        layout_right.addWidget(self.graphics, 1)
        layout_right.addWidget(self.spectra, 0)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_left, 0)
        layout.addLayout(layout_right, 1)

        self.setLayout(layout)

    def initializePage(self) -> None:
        self.drawTIC()

    def drawTIC(self) -> None:
        if self.image is not None:
            self.graphics.scene().removeItem(self.image)
        # imzml: ImzML = self.field("imzml")
        # self.imzml = ImzML(
        #     "/home/tom/Downloads/slide 8 at 19%.imzML",
        #     "/home/tom/Downloads/slide 8 at 19%.ibd",
        # )
        # sx, sy = self.imzml.image_size
        # px, py = self.imzml.pixel_size
        # rect = QtCore.QRectF(0, 0, sx * px, sy * py)
        # x = self.imzml.extract_masses(np.array([768.55]))[:, :, 0]
        #
        # x -= np.nanmin(x)
        # x /= np.nanmax(x)
        #
        self.image = ClickableImageItem.fromArray(
            x, rect, list(get_table(self.graphics.options.colortable))
        )
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

    def drawSpectraAtPos(self, pos: QtCore.QPoint) -> None:
        self.spectra.clear()

        # convert mapToData pos to stored spectrum pos
        px, py = pos.x() + 1, self.imzml.image_size[1] - pos.y()
        try:
            spectrum = self.imzml.spectra_map[(px, py)]
        except KeyError:
            return

        x = spectrum.get_binary_data(
            self.imzml.mz_reference, self.imzml.mz_dtype, self.imzml.external_binary
        )
        y = spectrum.get_binary_data(
            self.imzml.intensity_reference,
            self.imzml.intensity_dtype,
            self.imzml.external_binary,
        )
        self.spectra.drawCentroidSpectra(x, y)


app = QtWidgets.QApplication()
wiz = QtWidgets.QWizard()
wiz.addPage(ImzMLImportPage(Path("/home/tom/Downloads/slide 8 at 19%.imzML")))
wiz.addPage(ImzMLTargetMassPage())
wiz.show()
app.exec()
