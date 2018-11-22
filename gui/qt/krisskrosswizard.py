from PyQt5 import QtCore, QtGui, QtWidgets

from util.krisskross import KrissKrossData
from util.importer import importNpz, importAgilentBatch
from gui.qt.configdialog import ConfigForm

import os


class KrissKrossWizard(QtWidgets.QWizard):
    def __init__(self,  config, parent=None):
        super().__init__(parent)

        self.addPage(KrissKrossStartPage())
        self.addPage(KrissKrossImportPage())
        self.addPage(KrissKrossConfigPage(config))

        self.setWindowTitle("Kriss Kross Import Wizard")

        self.resize(540, 480)

    def accept(self):
        kkds = []
        config = self.field("config")
        # Get config
        if self.field("radio_numpy"):
            layer_dict = {}
            # Import each file
            for path in self.field("paths"):
                ld_group = importNpz(path)
                for ld in ld_group:
                    layer_dict.setdefault(ld.isotope, []).append(ld)
            # Transform for each isotope
            for isotope, layers in layer_dict.items():
                kkd = KrissKrossData(isotope=isotope, config=config,
                                     source=self.fields["paths"][0])
                kkd.fromLayers([layer.data for layer in layers])
                kkds.append(kkd)

        elif self.field("radio_agilent"):
            for path in self.field("paths"):
                # lds = importNpz(path, config)
                pass
        elif self.field("radio_csv"):
            for path in self.field("paths"):
                # lds = importNpz(path, config)
                pass

        super().accept()


class KrissKrossStartPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("Overview")
        label = QtWidgets.QLabel(
            "This wizard will import SRR-LA-ICP-MS data. To begin, select "
            "the type of data to import. You may then import, reorder and "
            "configure the imported data.")
        label.setWordWrap(True)

        mode_box = QtWidgets.QGroupBox("Data type", self)

        radio_numpy = QtWidgets.QRadioButton("&Numpy archives", self)
        radio_agilent = QtWidgets.QRadioButton("&Agilent batches", self)
        radio_csv = QtWidgets.QRadioButton("&CSV layers", self)
        radio_numpy.setChecked(True)
        # Todo redo csv imports
        radio_csv.setCheckable(False)

        self.registerField("radio_numpy", radio_numpy)
        self.registerField("radio_agilent", radio_numpy)
        self.registerField("radio_csv", radio_numpy)

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(radio_numpy)
        box_layout.addWidget(radio_agilent)
        box_layout.addWidget(radio_csv)
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
        button_file = QtWidgets.QPushButton("Open")
        button_file.clicked.connect(self.buttonAdd)
        button_dir = QtWidgets.QPushButton("Open All...")
        button_dir.clicked.connect(self.buttonAddAll)

        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(button_file)
        control_layout.addWidget(button_dir)
        control_widget = QtWidgets.QWidget(self)
        control_widget.setLayout(control_layout)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(dir_box)
        main_layout.addWidget(control_widget)
        self.setLayout(main_layout)

        self.registerField("paths", self, "paths")

    def buttonAdd(self):
        if self.field("radio_numpy"):
            paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Select Files", "" "(*.npz)")
        elif self.field("radio_agilent"):
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Batch", "", QtWidgets.QFileDialog.ShowDirsOnly)
            paths = [path]
        elif self.field("radio_csv"):
            paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Select Files", "" "(*.csv)")
        for path in paths:
            self.list.addItem(path)

    def buttonAddAll(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", "",
            QtWidgets.QFileDialog.DontUseNativeDialog)
        files = os.listdir(path)
        files.sort()

        if self.field("radio_numpy"):
            ext = '.npz'
        elif self.field("radio_agilent"):
            ext = '.b'
        elif self.field("radio_csv"):
            ext = '.csv'

        for f in files:
            if f.lower().endswith(ext):
                self.list.addItem(os.path.join(path, f))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete or \
           event.key() == QtCore.Qt.Key_Backspace:
            for item in self.list.selectedItems():
                self.list.takeItem(self.list.row(item))
        super().keyPressEvent(event)

    def isComplete(self):
        min_files = 2
        # if self.field("radio_numpy"):
        #     min_files = 1

        return self.list.count() >= min_files

    @QtCore.pyqtProperty(list)
    def paths(self):
        paths = []
        for i in range(0, self.list.count()):
            paths.append(self.list.item(i).text())
        return paths


class KrissKrossConfigPage(QtWidgets.QWizardPage):
    def __init__(self, config, parent=None):
        super().__init__(parent)

        self.config = config

        self.form = ConfigForm(self.config, parent=self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.form)
        self.setLayout(layout)

        self.registerField("config", self, "config")

    def validatePage(self):
        for k in self.config.keys():
            v = float(getattr(self.form, k).text())
            self.config[k] = v

    # @QtCore.pyqtProperty(dict)
    # def config(self):
    #     return self.config
