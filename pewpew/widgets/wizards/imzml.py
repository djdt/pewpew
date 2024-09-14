import logging
from pathlib import Path

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

        self.path_binary = PathSelectWidget(
            external_binary or Path(), "Binary Data", [".ibd"], "File"
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

    def guessBinaryPath(self, imzml: Path) -> None:
        if imzml.with_suffix(".ibd").exists() and self.path_binary.path == "":
            self.path_binary.addPath(imzml.with_suffix(".ibd"))

    def validatePage(self):
        imzml = ImzML(self.path.path, external_binary=self.path.path_binary)
        self.setField("imzml", imzml)

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

        self.item = ScaledImageItem(QtGui.QImage(), QtCore.QRectF())
        self.graphics.scene().addItem(self.item)

        self.spectra = SpectraView()

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addWidget(self.mass_list, 1)
        layout_left.addWidget(self.mass_width, 0)

        layout_right  = QtWidgets.QVBoxLayout()
        layout_right.addWidget(self.graphics, 1)
        layout_right.addWidget(self.spectra, 0)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_left, 0)
        layout.addLayout(layout_right, 1)

        self.setLayout(layout)
        self.initializePage()

    def initializePage(self) -> None:
        self.drawTIC()

    def drawTIC(self) -> None:
        # imzml: ImzML = self.field("imzml")
        self.imzml = ImzML(
            "/home/tom/Downloads/slide 8 at 19%.imzML",
            "/home/tom/Downloads/slide 8 at 19%.ibd",
        )
        sx, sy = self.imzml.image_size
        px, py = self.imzml.pixel_size
        rect = QtCore.QRectF(0, 0, sx * px, sy * py)
        self.graphics.scene().removeItem(self.item)
        x = self.imzml.extract_tic()
        x = (x - x.min()) / (x.max() - x.min())

        self.item = ScaledImageItem.fromArray(
            x, rect, list(get_table(self.graphics.options.colortable))
        )
        self.graphics.scene().addItem(self.item)
        self.graphics.zoomReset()


app = QtWidgets.QApplication()
page = ImzMLTargetMassPage()
page.show()
app.exec()
