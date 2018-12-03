from PyQt5 import QtCore, QtGui, QtWidgets
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


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, calibration, current_isotope, isotopes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.calibration = calibration

        # LIne edits
        self.lineedit_gradient = QtWidgets.QLineEdit()
        self.lineedit_gradient.setValidator(
            QtGui.QDoubleValidator(-1e10, 1e10, 4))
        self.lineedit_gradient.setPlaceholderText("1.0")
        self.lineedit_intercept = QtWidgets.QLineEdit()
        self.lineedit_intercept.setValidator(
            QtGui.QDoubleValidator(-1e10, 1e10, 4))
        self.lineedit_intercept.setPlaceholderText("0.0")
        self.lineedit_unit = QtWidgets.QLineEdit()
        self.lineedit_unit.setPlaceholderText("<None>")

        # Form layout for line edits
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Gradient:", self.lineedit_gradient)
        form_layout.addRow("Intercept:", self.lineedit_intercept)
        form_layout.addRow("Unit:", self.lineedit_unit)

        # Isotope combo
        self.combo_isotopes = QtWidgets.QComboBox()
        self.combo_isotopes.addItems(isotopes)
        self.previous_index = self.combo_isotopes.findText(current_isotope)
        self.combo_isotopes.setCurrentIndex(self.previous_index)
        self.combo_isotopes.currentIndexChanged.connect(self.comboChanged)

        # Dialog buttons
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok,
                                               self)
        buttonBox.accepted.connect(self.accept)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.combo_isotopes, 1, QtCore.Qt.AlignRight)
        main_layout.addWidget(buttonBox)
        self.setLayout(main_layout)

        self.updateLineEdits()

    def updateLineEdits(self):
        new = self.combo_isotopes.currentText()

        if new in self.calibration['gradients']:
            self.lineedit_gradient.setText(
                str(self.calibration['gradients'][new]))
        else:
            self.lineedit_gradient.clear()
        if new in self.calibration['intercepts']:
            self.lineedit_intercept.setText(
                str(self.calibration['intercepts'][new]))
        else:
            self.lineedit_intercept.clear()
        if new in self.calibration['units']:
            self.lineedit_unit.setText(
                str(self.calibration['units'][new]))
        else:
            self.lineedit_unit.clear()

    def updateCalibration(self, isotope):
        gradient = self.lineedit_gradient.text()
        intercept = self.lineedit_intercept.text()
        unit = self.lineedit_unit.text()

        if gradient == "" or float(gradient) == 1.0:
            self.calibration['gradients'].pop(isotope, None)
        else:
            self.calibration['gradients'][isotope] = float(gradient)
        if intercept == "" or float(intercept) == 0.0:
            self.calibration['intercepts'].pop(isotope, None)
        else:
            self.calibration['intercepts'][isotope] = float(intercept)
        if unit == "":
            self.calibration['units'].pop(isotope, None)
        else:
            self.calibration['units'][isotope] = unit

    def comboChanged(self):
        previous = self.combo_isotopes.itemText(self.previous_index)
        self.updateCalibration(previous)
        self.updateLineEdits()
        self.previous_index = self.combo_isotopes.currentIndex()

    def accept(self):
        self.updateCalibration(self.combo_isotopes.currentText())
        super().accept()


class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.config = config

        # Line edits
        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setPlaceholderText(str(config['spotsize']))
        self.lineedit_spotsize.setValidator(QtGui.QDoubleValidator(0, 1e3, 2))
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setPlaceholderText(str(config['speed']))
        self.lineedit_speed.setValidator(QtGui.QDoubleValidator(0, 1e3, 2))
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setPlaceholderText(str(config['scantime']))
        self.lineedit_scantime.setValidator(QtGui.QDoubleValidator(0, 1e3, 4))

        # Form layout for line edits
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Spotsize (μm):", self.lineedit_spotsize)
        form_layout.addRow("Speed (μm):", self.lineedit_speed)
        form_layout.addRow("Scantime (s):", self.lineedit_scantime)

        # self.form = ConfigForm(config, parent=self)
        # Checkbox
        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")
        # Ok button
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok,
                                               self)
        buttonBox.accepted.connect(self.accept)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.check_all)
        main_layout.addWidget(buttonBox)
        self.setLayout(main_layout)

    def accept(self):
        if self.lineedit_spotsize.text() != "":
            self.config['spotsize'] = float(self.lineedit_spotsize.text())
        if self.lineedit_speed.text() != "":
            self.config['speed'] = float(self.lineedit_speed.text())
        if self.lineedit_scantime.text() != "":
            self.config['scantime'] = float(self.lineedit_scantime.text())
        super().accept()


#############
#  Save As  #
#############


class ExportDialog(QtWidgets.QDialog):
    def __init__(self,
                 source,
                 current_isotope,
                 num_isotopes=1,
                 num_layers=1,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.current_isotope = current_isotope
        self.num_isotopes = num_isotopes
        self.num_layers = num_layers

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
        if self.num_isotopes < 2:
            self.check_isotopes.setEnabled(False)
        self.check_layers = QtWidgets.QCheckBox("Save all layers.")
        self.check_layers.stateChanged.connect(self.inputChanged)
        if self.num_layers < 2:
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
        filter = ("CSV files(*.csv);;Numpy archives(*.npz);;"
                  "PNG images(*.png);;")
        if self.num_layers > 1:
            filter += "Rectilinear VTKs(*.vtr);;"
        filter += "All files(*)"
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export As", self.lineedit_file.text(), filter,
            "CSV files(*.csv)", QtWidgets.QFileDialog.DontConfirmOverwrite)
        if path:
            self.lineedit_file.setText(path)
            self.inputChanged()

    def inputChanged(self):
        self.updateChecks()
        self.redrawPreview()

    def updateChecks(self):
        ext = os.path.splitext(self.lineedit_file.text())[1].lower()

        if ext == '.vtr':
            self.check_isotopes.setEnabled(False)
            self.check_isotopes.setChecked(False)
            self.check_layers.setEnabled(False)
            self.check_layers.setChecked(False)
        elif ext == '.npz':
            self.check_isotopes.setEnabled(False)
            self.check_isotopes.setChecked(False)
            self.check_layers.setEnabled(False)
            self.check_layers.setChecked(False)
        else:
            if self.num_isotopes > 1:
                self.check_isotopes.setEnabled(True)
            if self.num_layers > 1:
                self.check_layers.setEnabled(True)

    def redrawPreview(self):
        path = self.getPath(
            isotope=self.current_isotope
            if self.check_isotopes.isChecked() else None,
            layer=1 if self.check_layers.isChecked() else None)
        self.lineedit_preview.setText(os.path.basename(path))
