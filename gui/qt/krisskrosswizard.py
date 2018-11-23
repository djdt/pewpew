from PyQt5 import QtCore, QtGui, QtWidgets

from util.krisskross import KrissKrossData
from util.importer import importNpz, importAgilentBatch
from gui.qt.configdialog import ConfigForm

import os


class KrissKrossWizard(QtWidgets.QWizard):
    def __init__(self,  config, parent=None):
        super().__init__(parent)

        self.krisskrossdata = []

        self.addPage(KrissKrossStartPage())
        self.addPage(KrissKrossImportPage())
        self.addPage(KrissKrossConfigPage(config))

        self.setWindowTitle("Kriss Kross Import Wizard")

        self.resize(540, 480)

    def accept(self):
        config = self.field("config")
        layer_dict = {}
        if self.field("radio_numpy"):
            # Use the config from the first file
            ld_group = importNpz(self.field("paths")[0])
            config = ld_group[0].config
            for ld in ld_group:
                layer_dict.setdefault(ld.isotope, []).append(ld)
                # Import the rest of the files with overriden config
            for path in self.field("paths")[1:]:
                ld_group = importNpz(path, config)
                for ld in ld_group:
                    layer_dict.setdefault(ld.isotope, []).append(ld)

        elif self.field("radio_agilent"):
            for path in self.field("paths"):
                ld_group = importAgilentBatch(path, config)
                for ld in ld_group:
                    layer_dict.setdefault(ld.isotope, []).append(ld)

        elif self.field("radio_csv"):
            for path in self.field("paths"):
                # lds = importNpz(path, config)
                pass

        for isotope, layers in layer_dict.items():
            kkd = KrissKrossData(isotope=isotope, config=config,
                                 source=self.field("paths")[0])
            kkd.fromLayers([layer.data for layer in layers],
                           horizontal_first=self.field("check_horizontal"))
            self.krisskrossdata.append(kkd)

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
        radio_csv.setEnabled(False)

        self.registerField("radio_numpy", radio_numpy)
        self.registerField("radio_agilent", radio_agilent)
        self.registerField("radio_csv", radio_csv)

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
        check_horizontal = QtWidgets.QCheckBox("Horizontal first layer.")
        check_horizontal.setChecked(True)
        self.registerField("check_horizontal", check_horizontal)

        # Buttons
        button_file = QtWidgets.QPushButton("Open")
        button_file.clicked.connect(self.buttonAdd)
        button_dir = QtWidgets.QPushButton("Open All...")
        button_dir.clicked.connect(self.buttonAddAll)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(button_file)
        button_layout.addWidget(button_dir)
        button_box = QtWidgets.QWidget()
        button_box.setLayout(button_layout)

        # control_layout = QtWidgets.QVBoxLayout()
        # control_layout.addWidget(check_horizontal)
        # control_layout.addWidget(button_box)
        # control_widget = QtWidgets.QWidget(self)
        # control_widget.setLayout(control_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(dir_box)
        main_layout.addWidget(check_horizontal)
        main_layout.addWidget(button_box)
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
        if len(path) == 0:
            return
        files = os.listdir(path)
        files.sort()

        ext = '.npz'
        if self.field("radio_agilent"):
            ext = '.b'
        elif self.field("radio_csv"):
            ext = '.csv'

        print(self.field("radio_numpy"))
        print(self.field("radio_agilent"))
        print(self.field("radio_csv"))
        print(ext)

        for f in files:
            if f.lower().endswith(ext):
                print(f)
                self.list.addItem(os.path.join(path, f))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete or \
           event.key() == QtCore.Qt.Key_Backspace:
            for item in self.list.selectedItems():
                self.list.takeItem(self.list.row(item))
        super().keyPressEvent(event)

    def initializePage(self):
        # No need for config if numpy are used
        self.setFinalPage(self.field("radio_numpy"))

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
        return True

    # @QtCore.pyqtProperty(dict)
    # def config(self):
    #     return self.config
