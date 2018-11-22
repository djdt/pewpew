from PyQt5 import QtCore, QtGui, QtWidgets

from util.importer import importNpz, importAgilentBatch
from gui.qt.configdialog import ConfigForm


class KrissKrossWizard(QtWidgets.QWizard):
    def __init__(self,  config, parent=None):
        super().__init__(parent)

        self.config = config

        self.addPage(KrissKrossStartPage())
        self.addPage(KrissKrossImportPage())
        # self.addPage(KrissKrossConfigPage(config))

        self.setWindowTitle("Kriss Kross Import Wizard")

    def accept(self):
        pass


class KrissKrossStartPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("Overview")
        label = QtWidgets.QLabel(
            "This wizard will import SRR-LA-ICP-MS data. To begin, select "
            "the type of data to import. You may then import, reorder and "
            "configure the imported data.")

        mode_box = QtWidgets.QGroupBox("Data type", self)

        radio_numpy = QtWidgets.QRadioButton("&Numpy archives", self)
        radio_agilent = QtWidgets.QRadioButton("&Agilent batches", self)
        radio_csv = QtWidgets.QRadioButton("&CSV layers", self)
        radio_numpy.setChecked(True)

        box_layout = QtWidgets.QVBoxLayout()
        box_layout.addWidget(radio_numpy)
        box_layout.addWidget(radio_agilent)
        box_layout.addWidget(radio_csv)
        mode_box.setLayout(box_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(label)
        main_layout.addWidget(mode_box)
        self.setLayout(main_layout)

        self.registerField("radio_numpy", radio_numpy)
        self.registerField("radio_agilent", radio_numpy)
        self.registerField("radio_csv", radio_numpy)

#     def inirializePage(self):
#         if self.field("radio_agilent"):
#             self.radio_agilent.setChecked(True)
#         elif self.field("radio_csv"):
#             self.radio_csv.setChecked(True)
#         else:
#             self.radio_numpy.setChecked(True)


class KrissKrossImportPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("Data")
        self.setSubTitle("Select and order files.")

        label = QtWidgets.QLabel("Directories:")
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.list.setTextElideMode(QtCore.Qt.ElideLeft)
        self.list.model().rowsInserted.connect(self.completeChanged)
        self.list.model().rowsRemoved.connect(self.completeChanged)

        button = QtWidgets.QPushButton("Open")
        button.clicked.connect(self.buttonAdd)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.list)
        layout.addWidget(button)
        self.setLayout(layout)

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

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete or \
           event.key() == QtCore.Qt.Key_Backspace:
            for item in self.list.selectedItems():
                self.list.takeItem(self.list.row(item))
        super().keyPressEvent(event)

    def isComplete(self):
        min_files = 2
        if self.field("radio_numpy"):
            min_files = 1

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

        form = ConfigForm(self.config, self)
        for k in self.config.keys():
            self.registerField(k, getattr(self.form, k))
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(form)
        self.setLayout(layout)
