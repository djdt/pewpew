from PyQt5 import QtGui, QtWidgets
import os.path

#######################
#  Colorrange dialog  #
#######################


class IntOrPercentValidator(QtGui.QIntValidator):
    def __init__(self, min_int=None, max_int=None, parent=None):
        super().__init__(parent)
        self.min_int = min_int
        self.max_int = max_int

    def validate(self, input, pos):
        if len(input) == 0:
            return (QtGui.QValidator.Intermediate, input, pos)

        if input.endswith('%'):
            if input.count('%') > 1:
                return (QtGui.QValidator.Invalid, input, pos)
            min_int = 0
            max_int = 100
        else:
            min_int = self.min_int
            max_int = self.max_int

        try:
            i = int(input.rstrip('%'))
        except ValueError:
            return (QtGui.QValidator.Invalid, input, pos)

        if min_int is not None and i < min_int:
            return (QtGui.QValidator.Intermediate, input, pos)

        if max_int is not None and i > max_int:
            return (QtGui.QValidator.Invalid, input, pos)

        return (QtGui.QValidator.Acceptable, input, pos)


class ColorRangeDialog(QtWidgets.QDialog):
    def __init__(self, current_range, parent=None):
        self.range = current_range
        super().__init__(parent)
        self.setWindowTitle("Colormap Range")

        self.lineedit_min = QtWidgets.QLineEdit()
        self.lineedit_min.setPlaceholderText(str(current_range[0]))
        self.lineedit_min.setToolTip("Enter absolute value or percentile.")
        self.lineedit_min.setValidator(
            IntOrPercentValidator(min_int=0, parent=self))
        self.lineedit_max = QtWidgets.QLineEdit()
        self.lineedit_max.setPlaceholderText(str(current_range[1]))
        self.lineedit_min.setValidator(
            IntOrPercentValidator(min_int=0, parent=self))
        self.lineedit_max.setToolTip("Enter absolute value or percentile.")

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Minimum:", self.lineedit_min)
        form_layout.addRow("Maximum:", self.lineedit_max)

        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel, self)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        main_layout = QtWidgets.QVBoxLayout()

        main_layout.addLayout(form_layout)
        main_layout.addWidget(buttonBox)
        self.setLayout(main_layout)

    def getRangeAsFloatOrPercent(self):
        minimum = self.lineedit_min.text()
        if len(minimum) == 0:
            minimum = self.range[0]
        elif not minimum.endswith('%'):
            minimum = int(minimum)
        maximum = self.lineedit_max.text()
        if len(maximum) == 0:
            maximum = self.range[1]
        elif not maximum.endswith('%'):
            maximum = int(maximum)

        return (minimum, maximum)


############
#  Config  #
############


class ConfigForm(QtWidgets.QGroupBox):
    def __init__(self, config, title=None, parent=None):
        super().__init__(title, parent)

        self.config = config

        layout = QtWidgets.QFormLayout()
        for k, v in self.config.items():
            le = QtWidgets.QLineEdit()
            le.setPlaceholderText(str(v))
            if k in ["gradient", "intercept"]:
                le.setValidator(QtGui.QDoubleValidator(-1e10, 1e10, 8))
            else:
                le.setValidator(QtGui.QDoubleValidator(0, 1e3, 4))
            layout.addRow(k.capitalize() + ":", le)
            setattr(self, k, le)

        self.setLayout(layout)


class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")

        main_layout = QtWidgets.QVBoxLayout()
        # Form layout for line edits
        self.form = ConfigForm(config, parent=self)
        # Checkbox
        self.check_all = QtWidgets.QCheckBox("Apply configs to all images.")
        # Ok button
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok,
                                               self)
        buttonBox.accepted.connect(self.accept)

        main_layout.addWidget(self.form)
        main_layout.addWidget(self.check_all)
        main_layout.addWidget(buttonBox)
        self.setLayout(main_layout)

        self.resize(480, 320)

    def accept(self):
        for k in self.form.config.keys():
            v = getattr(self.form, k).text()
            if v is not "":
                self.form.config[k] = float(v)
        super().accept()


#############
#  Save As  #
#############


class ExportDialog(QtWidgets.QDialog):
    def __init__(self, source, current_isotope, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exporter")
        self.current_isotope = current_isotope

        default_name = os.path.splitext(source)[0] + ".csv"

        self.lineedit_file = QtWidgets.QLineEdit(default_name)
        self.lineedit_file.setMinimumWidth(300)
        self.lineedit_file.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.Minimum)
        self.lineedit_file.textEdited.connect(self.inputChanged)
        button_file = QtWidgets.QPushButton("Select...")
        button_file.pressed.connect(self.onButtonFile)

        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(self.lineedit_file)
        file_layout.addWidget(button_file)

        self.check_isotopes = QtWidgets.QCheckBox("Save all isotopes.")
        self.check_isotopes.stateChanged.connect(self.inputChanged)
        self.check_layers = QtWidgets.QCheckBox("Save all layers.")
        self.check_layers.stateChanged.connect(self.inputChanged)
        self.check_layers.setEnabled(False)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save
            | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        label_preview = QtWidgets.QLabel("Preview: ")
        self.lineedit_preview = QtWidgets.QLineEdit()
        self.lineedit_preview.setEnabled(False)
        self.redrawPreview()

        preview_layout = QtWidgets.QHBoxLayout()
        preview_layout.addWidget(label_preview)
        preview_layout.addWidget(self.lineedit_preview)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(file_layout)
        main_layout.addWidget(self.check_isotopes)
        main_layout.addWidget(self.check_layers)
        main_layout.addLayout(preview_layout)
        main_layout.addWidget(button_box)
        self.setLayout(main_layout)

    def getPath(self, isotope=None, layer=None):
        path, ext = os.path.splitext(self.lineedit_file.text())
        if isotope is not None:
            path += f"_{isotope}"
        if layer is not None:
            path += f"_layer{layer}"
        return path + ext

    def onButtonFile(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export As", self.lineedit_file.text(), "CSV files(*.csv);;"
            "PNG images(*.png);;All files(*)", "CSV files(*.csv)",
            QtWidgets.QFileDialog.DontConfirmOverwrite)
        if path:
            self.lineedit_file.setText(path)

    def optionsChanged(self):
        self.updateChecks()
        self.redrawPreview()

    def updateChecks(self):
        ext = os.path.splitext(self.lineedit_file.text())[1].lower()

        if ext == '.vtr':
            self.check_layers.setEnabled(False)
            self.check_isotopes.setEnabled(False)
            self.check_layers.setChecked(False)
            self.check_isotopes.setChecked(False)
        elif ext == '.npz':
            self.check_layers.setEnabled(False)
            self.check_isotopes.setEnabled(False)
            self.check_layers.setChecked(False)
            self.check_isotopes.setChecked(False)
        else:
            self.check_layers.setEnabled(True)
            self.check_isotopes.setEnabled(True)

        self.redrawPreview()

    def redrawPreview(self):
        path = self.getPath(
            isotope=self.current_isotope
            if self.check_isotopes.isChecked() else None,
            layer=1 if self.check_layers.isChecked() else None)
        self.lineedit_preview.setText(os.path.basename(path))
