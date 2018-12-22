from PyQt5 import QtCore, QtGui, QtWidgets
import os.path

from gui.validators import PercentValidator


class ApplyDialog(QtWidgets.QDialog):

    applyPressed = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply,
            self,
        )
        self.button_box.clicked.connect(self.buttonClicked)

    def buttonClicked(self, button):
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.Apply:
            self.apply()
            self.applyPressed.emit(self)
        elif sb == QtWidgets.QDialogButtonBox.Ok:
            self.accept()
        else:
            self.reject()

    def apply(self):
        pass


class ColorRangeDialog(ApplyDialog):
    def __init__(self, current_range, parent=None):
        self.range = current_range
        super().__init__(parent)
        self.setWindowTitle("Colormap Range")

        self.lineedit_min = QtWidgets.QLineEdit()
        self.lineedit_min.setPlaceholderText(str(current_range[0]))
        self.lineedit_min.setToolTip("Percentile for minium colormap value.")
        self.lineedit_min.setValidator(PercentValidator(parent=self.lineedit_min))
        self.lineedit_max = QtWidgets.QLineEdit()
        self.lineedit_max.setPlaceholderText(str(current_range[1]))
        self.lineedit_min.setValidator(PercentValidator(parent=self.lineedit_max))
        self.lineedit_max.setToolTip("Percentile for maximum colormap value.")

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Minimum:", self.lineedit_min)
        form_layout.addRow("Maximum:", self.lineedit_max)

        main_layout = QtWidgets.QVBoxLayout()

        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def apply(self):
        min_text, max_text = self.lineedit_min.text(), self.lineedit_max.text()
        cmap_range = [
            min_text if min_text != "" else self.range[0],
            max_text if max_text != "" else self.range[1],
        ]
        self.parent().viewconfig["cmap_range"] = cmap_range
        self.parent().draw()

    def accept(self):
        min_text, max_text = self.lineedit_min.text(), self.lineedit_max.text()
        if min_text != "":
            self.range[0] = min_text
        if max_text != "":
            self.range[1] = max_text
        super().accept()


class CalibrationDialog(ApplyDialog):
    def __init__(self, calibration, current_isotope, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.calibration = calibration

        # LIne edits
        self.lineedit_gradient = QtWidgets.QLineEdit()
        self.lineedit_gradient.setValidator(QtGui.QDoubleValidator(-1e10, 1e10, 4))
        self.lineedit_gradient.setPlaceholderText("1.0")
        self.lineedit_intercept = QtWidgets.QLineEdit()
        self.lineedit_intercept.setValidator(QtGui.QDoubleValidator(-1e10, 1e10, 4))
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
        self.combo_isotopes.addItems([k for k in self.calibration.keys()])
        self.combo_isotopes.setCurrentText(current_isotope)
        self.previous_index = self.combo_isotopes.currentIndex()
        self.combo_isotopes.currentIndexChanged.connect(self.comboChanged)

        # Dialog buttons
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.combo_isotopes, 1, QtCore.Qt.AlignRight)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

        self.updateLineEdits()

    def updateLineEdits(self):
        isotope = self.combo_isotopes.currentText()

        gradient = self.calibration[isotope]["gradient"]
        if gradient == 1.0:
            self.lineedit_gradient.clear()
        else:
            self.lineedit_gradient.setText(str(gradient))
        intercept = self.calibration[isotope]["intercept"]
        if intercept == 0.0:
            self.lineedit_intercept.clear()
        else:
            self.lineedit_intercept.setText(str(intercept))
        unit = self.calibration[isotope]["unit"]
        if unit is None:
            self.lineedit_unit.clear()
        else:
            self.lineedit_unit.setText(str(unit))

    def updateCalibration(self, isotope):
        gradient = self.lineedit_gradient.text()
        intercept = self.lineedit_intercept.text()
        unit = self.lineedit_unit.text()

        if gradient != "":
            self.calibration[isotope]["gradient"] = float(gradient)
        if intercept != "":
            self.calibration[isotope]["intercept"] = float(intercept)
        if unit != "":
            self.calibration[isotope]["unit"] = unit

    def comboChanged(self):
        previous = self.combo_isotopes.itemText(self.previous_index)
        self.updateCalibration(previous)
        self.updateLineEdits()
        self.previous_index = self.combo_isotopes.currentIndex()

    def apply(self):
        self.updateCalibration(self.combo_isotopes.currentText())

    def accept(self):
        self.updateCalibration(self.combo_isotopes.currentText())
        super().accept()


class ConfigDialog(ApplyDialog):
    def __init__(self, config, apply_func=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.spotsize = config["spotsize"]
        self.speed = config["speed"]
        self.scantime = config["scantime"]

        # Line edits
        self.lineedit_spotsize = QtWidgets.QLineEdit()
        self.lineedit_spotsize.setPlaceholderText(str(self.spotsize))
        self.lineedit_spotsize.setValidator(QtGui.QDoubleValidator(0, 1e3, 2))
        self.lineedit_speed = QtWidgets.QLineEdit()
        self.lineedit_speed.setPlaceholderText(str(self.speed))
        self.lineedit_speed.setValidator(QtGui.QDoubleValidator(0, 1e3, 2))
        self.lineedit_scantime = QtWidgets.QLineEdit()
        self.lineedit_scantime.setPlaceholderText(str(self.scantime))
        self.lineedit_scantime.setValidator(QtGui.QDoubleValidator(0, 1e3, 4))

        # Form layout for line edits
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Spotsize (μm):", self.lineedit_spotsize)
        form_layout.addRow("Speed (μm):", self.lineedit_speed)
        form_layout.addRow("Scantime (s):", self.lineedit_scantime)

        # Checkbox
        self.check_all = QtWidgets.QCheckBox("Apply config to all images.")
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.check_all)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def updateConfig(self):
        if self.lineedit_spotsize.text() != "":
            self.spotsize = float(self.lineedit_spotsize.text())
        if self.lineedit_speed.text() != "":
            self.speed = float(self.lineedit_speed.text())
        if self.lineedit_scantime.text() != "":
            self.scantime = float(self.lineedit_scantime.text())

    def apply(self):
        self.updateConfig()

    def accept(self):
        self.updateConfig()
        super().accept()


class TrimDialog(ApplyDialog):
    def __init__(self, trim=(0, 0), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trim")
        self.trim = trim

        self.lineedit_left = QtWidgets.QLineEdit()
        self.lineedit_left.setPlaceholderText(str(trim[0]))
        self.lineedit_left.setValidator(QtGui.QIntValidator(0, 1e9))
        self.lineedit_right = QtWidgets.QLineEdit()
        self.lineedit_right.setPlaceholderText(str(trim[1]))
        self.lineedit_right.setValidator(QtGui.QIntValidator(0, 1e9))

        self.combo_trim = QtWidgets.QComboBox()
        self.combo_trim.addItems(["rows", "s", "μm"])
        self.combo_trim.setCurrentIndex(1)
        self.combo_trim.currentIndexChanged.connect(self.comboTrim)

        layout_trim = QtWidgets.QHBoxLayout()
        layout_trim.addWidget(QtWidgets.QLabel("Left:"))
        layout_trim.addWidget(self.lineedit_left)
        layout_trim.addWidget(QtWidgets.QLabel("Right:"))
        layout_trim.addWidget(self.lineedit_right)
        layout_trim.addWidget(self.combo_trim)

        self.check_all = QtWidgets.QCheckBox("Apply trim to all images.")

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_trim)
        layout_main.addWidget(self.check_all)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

    def comboTrim(self):
        if self.combo_trim.currentText() == "rows":
            self.lineedit_left.setValidator(QtGui.QIntValidator(0, 1e9))
            self.lineedit_right.setValidator(QtGui.QIntValidator(0, 1e9))
        else:
            self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
            self.lineedit_right.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))

    def updateTrim(self):
        if self.lineedit_left.text() != "":
            self.trim[0] = float(self.lineedit_left.text())
        if self.lineedit_right.text() != "":
            self.trim[1] = float(self.lineedit_right.text())

    def apply(self):
        self.updateTrim()

    def accept(self):
        self.updateTrim()
        super().accept()


class ExportDialog(QtWidgets.QDialog):
    def __init__(self, lasers, default_path="", default_isotope=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Data")
        self.names = [laser.name for laser in lasers]
        self.isotopes = []
        for laser in lasers:
            self.isotopes.extend(i for i in laser.isotopes() if i not in self.isotopes)
        self.default_isotope = default_isotope

        self.layers = max(laser.layers() for laser in lasers)

        self.lineedit_path = QtWidgets.QLineEdit(default_path)
        self.button_path = QtWidgets.QPushButton("Select")

        self.check_isotopes = QtWidgets.QCheckBox("Export all isotopes.")
        self.check_layers = QtWidgets.QCheckBox("Export all layers.")

        self.combo_isotopes = QtWidgets.QComboBox()

        self.lineedit_preview = QtWidgets.QLineEdit()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )

        self.initialiseWidgets()
        self.layoutWidgets()

        self.contextEnable()
        self.drawPreview()

    def initialiseWidgets(self):
        self.lineedit_path.setMinimumWidth(300)
        self.lineedit_path.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum
        )
        self.lineedit_path.textEdited.connect(self.changed)

        self.button_path.pressed.connect(self.buttonPath)

        self.check_isotopes.stateChanged.connect(self.changed)
        self.check_layers.stateChanged.connect(self.changed)

        self.combo_isotopes.addItems(self.isotopes)
        self.combo_isotopes.currentIndexChanged.connect(self.changed)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.lineedit_preview = QtWidgets.QLineEdit()
        self.lineedit_preview.setEnabled(False)

    def layoutWidgets(self):
        layout_path = QtWidgets.QHBoxLayout()
        layout_path.addWidget(QtWidgets.QLabel("Basename:"))
        layout_path.addWidget(self.lineedit_path)
        layout_path.addWidget(self.button_path)

        layout_isotopes = QtWidgets.QHBoxLayout()
        layout_isotopes.addWidget(self.check_isotopes)
        layout_isotopes.addWidget(self.combo_isotopes)

        layout_preview = QtWidgets.QHBoxLayout()
        layout_preview.addWidget(QtWidgets.QLabel("Preview:"))
        layout_preview.addWidget(self.lineedit_preview)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_path)
        layout_main.addLayout(layout_isotopes)
        layout_main.addWidget(self.check_layers)
        layout_main.addLayout(layout_preview)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

    def buttonPath(self):
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export",
            self.lineedit_path.text(),
            "CSV files(*.csv);;Numpy archives(*.npz);;PNG images(*.png);;"
            "Rectilinear VTKs(*.vtr);;All files(*)",
            "All files(*)",
            QtWidgets.QFileDialog.DontConfirmOverwrite,
        )
        if path:
            self.lineedit_path.setText(path)
            self.changed()

    def changed(self):
        self.contextEnable()
        self.drawPreview()

    def contextEnable(self):
        ext = os.path.splitext(self.lineedit_path.text())[1].lower()

        if len(self.isotopes) == 1:
            self.check_isotopes.setEnabled(False)
            self.check_isotopes.setChecked(False)
        elif ext in [".npz", ".vtr"]:
            if self.check_isotopes.isEnabled():
                self.check_isotopes.setEnabled(False)
                self.check_isotopes.setChecked(True)
        else:
            self.check_isotopes.setEnabled(True)

        self.combo_isotopes.setEnabled(not self.check_isotopes.isChecked())

        if self.layers == 1:
            self.check_layers.setEnabled(False)
            self.check_layers.setChecked(False)
        elif ext in [".npz", ".vtr"]:
            if self.check_layers.isEnabled():
                self.check_layers.setEnabled(False)
                self.check_layers.setChecked(True)
        else:
            self.check_layers.setEnabled(True)

    def drawPreview(self):
        path = self.getPath(
            self.names[0] if len(self.names) > 1 else None,
            isotope=self.combo_isotopes.currentText()
            if self.check_isotopes.isEnabled()
            else None,
            layer=1 if self.check_layers.isChecked() else None,
        )

        if not os.path.isdir(path):
            path = os.path.basename(path)
        self.lineedit_preview.setText(path)

    def getPath(self, name=None, isotope=None, layer=None):
        path, ext = os.path.splitext(self.lineedit_path.text())
        if name is not None:
            path += f"_{name}"
        if isotope is not None:
            path += f"_{isotope}"
        if layer is not None:
            path += f"_{layer}"
        return path + ext
