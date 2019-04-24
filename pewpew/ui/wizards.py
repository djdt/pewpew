import os
from fractions import Fraction

from PyQt5 import QtCore, QtWidgets

from pewpew.ui.widgets.multipledirdialog import MultipleDirDialog
from pewpew.ui.validators import DecimalValidator

from laserlib.krisskross import KrissKross, KrissKrossConfig
from pewpew.lib import io

from typing import List


class KrissKrossWizard(QtWidgets.QWizard):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        config = KrissKrossConfig()

        self.data: KrissKross = KrissKross()

        self.addPage(KrissKrossStartPage())
        self.addPage(KrissKrossImportPage())
        self.addPage(KrissKrossConfigPage(config))

        self.setWindowTitle("Kriss Kross Import Wizard")

        self.resize(540, 480)

    def accept(self) -> None:
        config = self.field("config")
        paths = self.field("paths")
        lasers = []

        if self.field("radio_numpy"):
            for path in paths:
                lds = io.npz.load(path)
                if len(lds) > 1:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Import Error",
                        f'Archive "{os.path.basename(path)}" '
                        "contains more than one image.",
                    )
                    return
                lasers.append(lds[0])
            # Read the config from the frist file
            config = self.layers[0].config
        elif self.field("radio_agilent"):
            for path in paths:
                lasers.append(io.agilent.load(path, config))
        elif self.field("radio_thermo"):
            for path in paths:
                lasers.append(io.thermo.load(path, config))

        self.data = KrissKross(
            [laser.data for laser in lasers],
            config=config,
            calibration=lasers[0].calibration,
            name=os.path.splitext(os.path.basename(paths[0]))[0],
            filepath=paths[0],
        )

        super().accept()


class KrissKrossStartPage(QtWidgets.QWizardPage):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.setTitle("Overview")
        label = QtWidgets.QLabel(
            "This wizard will import SRR-LA-ICP-MS data. To begin, select "
            "the type of data to import. You may then import, reorder and "
            "configure the imported data. Once imported KrissKross image "
            "configurations cannot be changed."
        )
        label.setWordWrap(True)

        mode_box = QtWidgets.QGroupBox("Data type", self)

        radio_numpy = QtWidgets.QRadioButton("&Numpy archives", self)
        radio_agilent = QtWidgets.QRadioButton("&Agilent batches", self)
        radio_thermo = QtWidgets.QRadioButton("&Thermo iCap CSV exports", self)
        radio_numpy.setChecked(True)

        self.registerField("radio_numpy", radio_numpy)
        self.registerField("radio_agilent", radio_agilent)
        self.registerField("radio_thermo", radio_thermo)

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(radio_numpy)
        box_layout.addWidget(radio_agilent)
        box_layout.addWidget(radio_thermo)
        mode_box.setLayout(box_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(label)
        main_layout.addWidget(mode_box)
        self.setLayout(main_layout)


class KrissKrossImportPage(QtWidgets.QWizardPage):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.setTitle("Files and Directories")
        # List and box
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.list.setTextElideMode(QtCore.Qt.ElideLeft)
        self.list.model().rowsInserted.connect(self.completeChanged)
        self.list.model().rowsRemoved.connect(self.completeChanged)

        dir_box = QtWidgets.QGroupBox("Layer Order", self)
        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(self.list)
        dir_box.setLayout(box_layout)

        # Buttons
        button_file = QtWidgets.QPushButton("Open")
        button_file.clicked.connect(self.buttonAdd)
        button_dir = QtWidgets.QPushButton("Open All...")
        button_dir.clicked.connect(self.buttonAddAll)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(button_file)
        button_layout.addWidget(button_dir)
        box_layout.addLayout(button_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(dir_box)
        self.setLayout(main_layout)

        self.registerField("paths", self, "paths")

    def buttonAdd(self) -> None:
        if self.field("radio_numpy"):
            paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Select Files", "" "Numpy archives(*.npz);;All files(*)"
            )
        elif self.field("radio_agilent"):
            paths = MultipleDirDialog.getExistingDirectories(self, "Select Batches", "")
        elif self.field("radio_thermo"):
            paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Select Files", "" "CSV files(*.csv);;All files(*)"
            )

        for path in paths:
            self.list.addItem(path)

    def buttonAddAll(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if len(path) == 0:
            return
        files = os.listdir(path)
        files.sort()

        ext = ".npz"
        if self.field("radio_agilent"):
            ext = ".b"
        elif self.field("radio_thermo"):
            ext = ".csv"

        for f in files:
            if f.lower().endswith(ext):
                self.list.addItem(os.path.join(path, f))

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if (
            event.key() == QtCore.Qt.Key_Delete
            or event.key() == QtCore.Qt.Key_Backspace
        ):
            for item in self.list.selectedItems():
                self.list.takeItem(self.list.row(item))
        super().keyPressEvent(event)

    # def initializePage(self) -> None:
    #     # No need for config if numpy are used
    #     self.setFinalPage(self.field("radio_numpy"))

    def isComplete(self) -> bool:
        return self.list.count() >= 2

    @QtCore.pyqtProperty(list)
    def paths(self) -> List[str]:
        paths = []
        for i in range(0, self.list.count()):
            paths.append(self.list.item(i).text())
        return paths


class KrissKrossConfigPage(QtWidgets.QWizardPage):
    def __init__(self, config: KrissKrossConfig, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.dconfig = config

        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setPlaceholderText(str(config.spotsize))
        self.lineedit_spotsize.setValidator(DecimalValidator(0, 1e3, 4))
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setPlaceholderText(str(config.speed))
        self.lineedit_speed.setValidator(DecimalValidator(0, 1e3, 4))
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setPlaceholderText(str(config.scantime))
        self.lineedit_scantime.setValidator(DecimalValidator(0, 1e3, 4))

        # Krisskross params
        self.lineedit_warmup = QtWidgets.QLineEdit()
        self.lineedit_warmup.setPlaceholderText(str(config.warmup))
        self.lineedit_warmup.setValidator(DecimalValidator(0, 1e2, 2))
        self.spinbox_offsets = QtWidgets.QSpinBox()
        self.spinbox_offsets.setRange(2, 10)
        self.spinbox_offsets.setValue(
            config.subpixel_gcd.limit_denominator().denominator
        )
        self.spinbox_offsets.setToolTip(
            "The number of subpixels per pixel in each dimension."
        )

        # Form layout for line edits
        config_layout = QtWidgets.QFormLayout()
        config_layout.addRow("Spotsize (μm):", self.lineedit_spotsize)
        config_layout.addRow("Speed (μm):", self.lineedit_speed)
        config_layout.addRow("Scantime (s):", self.lineedit_scantime)

        config_gbox = QtWidgets.QGroupBox("Laser Configuration", self)
        config_gbox.setLayout(config_layout)

        params_layout = QtWidgets.QFormLayout()
        params_layout.addRow("Warmup (s):", self.lineedit_warmup)
        params_layout.addRow("Subpixel width:", self.spinbox_offsets)

        params_gbox = QtWidgets.QGroupBox("KrissKross Parameters", self)
        params_gbox.setLayout(params_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(config_gbox)
        layout.addWidget(params_gbox)
        self.setLayout(layout)

        self.registerField("config", self, "config")

    @QtCore.pyqtProperty(QtCore.QVariant)
    def config(self) -> KrissKrossConfig:
        if self.lineedit_spotsize.text() != "":
            self.dconfig.spotsize = float(self.lineedit_spotsize.text())
        if self.lineedit_speed.text() != "":
            self.dconfig.speed = float(self.lineedit_speed.text())
        if self.lineedit_scantime.text() != "":
            self.dconfig.scantime = float(self.lineedit_scantime.text())
        if self.lineedit_warmup.text() != "":
            self.dconfig.warmup = float(self.lineedit_warmup.text())

        v = self.spinbox_offsets.value()
        self.dconfig.pixel_offsets = [Fraction(i, v) for i in range(0, v)]
        self.dconfig._calculate_subpixel_params()

        return self.dconfig
