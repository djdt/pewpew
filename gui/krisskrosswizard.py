from PyQt5 import QtCore, QtGui, QtWidgets

from util.krisskross import KrissKrossData
from util.importer import importNpz, importAgilentBatch

import os


class KrissKrossWizard(QtWidgets.QWizard):
    def __init__(self, config, parent=None):
        super().__init__(parent)

        self.data = None

        self.addPage(KrissKrossStartPage())
        self.addPage(KrissKrossImportPage())
        self.addPage(KrissKrossConfigPage(config))

        self.setWindowTitle("Kriss Kross Import Wizard")

        self.resize(540, 480)

    def accept(self):
        config = self.field("config")
        calibration = None
        paths = self.field("paths")
        layers = []

        if self.field("radio_numpy"):
            for path in paths:
                lds = importNpz(path)
                if len(lds) > 1:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Import Error",
                        f'Archive "{os.path.basename(path)}" '
                        "contains more than one image.",
                    )
                    return
                layers.append(lds[0])
            # Read the config and calibration from the frist file
            config = self.layers[0].config
            calibration = self.layers[0].calibration
        elif self.field("radio_agilent"):
            for path in paths:
                layers.append(importAgilentBatch(path, None))
        elif self.field("radio_thermo"):
            for path in paths:
                layers.append(importAgilentBatch(path, None))

        self.data = KrissKrossData(
            None,
            config=config,
            calibration=calibration,
            name=os.path.splitext(os.path.basename(paths[0]))[1],
            source=paths[0],
        )
        self.data.fromLayers(
            [layer.data for layer in layers],
            warmup_time=float(self.field("lineedit_warmup")),
            horizontal_first=self.field("check_horizontal"),
        )

        super().accept()


class KrissKrossStartPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
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
    def __init__(self, parent=None):
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
        # Controls
        check_horizontal = QtWidgets.QCheckBox("Horizontal first layer.")
        check_horizontal.setChecked(True)
        self.registerField("check_horizontal", check_horizontal)

        self.lineedit_warmup = QtWidgets.QLineEdit(str(13.0))
        self.lineedit_warmup.setValidator(QtGui.QDoubleValidator(0, 1e2, 2))
        self.lineedit_warmup.textChanged.connect(self.completeChanged)
        self.registerField("lineedit_warmup", self.lineedit_warmup)
        warmup_layout = QtWidgets.QHBoxLayout()
        warmup_layout.addWidget(QtWidgets.QLabel("Warmup (s):"))
        warmup_layout.addWidget(self.lineedit_warmup)

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
        main_layout.addWidget(check_horizontal)
        main_layout.addLayout(warmup_layout)
        self.setLayout(main_layout)

        self.registerField("paths", self, "paths")

    def buttonAdd(self):
        if self.field("radio_numpy"):
            paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Select Files", "" "Numpy archives(*.npz);;All files(*)"
            )
        elif self.field("radio_agilent"):
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Batch", "", QtWidgets.QFileDialog.ShowDirsOnly
            )
            if len(path) == 0:
                return
            paths = [path]
        elif self.field("radio_thermo"):
            paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Select Files", "" "CSV files(*.csv);;All files(*)"
            )

        for path in paths:
            self.list.addItem(path)

    def buttonAddAll(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", "", QtWidgets.QFileDialog.DontUseNativeDialog
        )
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

    def keyPressEvent(self, event):
        if (
            event.key() == QtCore.Qt.Key_Delete
            or event.key() == QtCore.Qt.Key_Backspace
        ):
            for item in self.list.selectedItems():
                self.list.takeItem(self.list.row(item))
        super().keyPressEvent(event)

    def initializePage(self):
        # No need for config if numpy are used
        self.setFinalPage(self.field("radio_numpy"))

    def isComplete(self):
        return self.list.count() >= 2 and len(self.lineedit_warmup.text()) > 0

    @QtCore.pyqtProperty(list)
    def paths(self):
        paths = []
        for i in range(0, self.list.count()):
            paths.append(self.list.item(i).text())
        return paths


class KrissKrossConfigPage(QtWidgets.QWizardPage):
    def __init__(self, config, parent=None):
        super().__init__(parent)

        self.dconfig = config

        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setPlaceholderText(str(config["spotsize"]))
        self.lineedit_spotsize.setValidator(QtGui.QDoubleValidator(0, 1e3, 4))
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setPlaceholderText(str(config["speed"]))
        self.lineedit_speed.setValidator(QtGui.QDoubleValidator(0, 1e3, 4))
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setPlaceholderText(str(config["scantime"]))
        self.lineedit_scantime.setValidator(QtGui.QDoubleValidator(0, 1e3, 4))

        # Form layout for line edits
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Spotsize (μm):", self.lineedit_spotsize)
        form_layout.addRow("Speed (μm):", self.lineedit_speed)
        form_layout.addRow("Scantime (s):", self.lineedit_scantime)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form_layout)
        self.setLayout(layout)

        self.registerField("config", self, "config")

    @QtCore.pyqtProperty(QtCore.QVariant)
    def config(self):
        if self.lineedit_spotsize.text() != "":
            self.dconfig["spotsize"] = float(self.lineedit_spotsize.text())
        if self.lineedit_speed.text() != "":
            self.dconfig["speed"] = float(self.lineedit_speed.text())
        if self.lineedit_scantime.text() != "":
            self.dconfig["scantime"] = float(self.lineedit_scantime.text())
        return self.dconfig
