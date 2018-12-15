from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from matplotlib.lines import Line2D

from gui.canvas import Canvas

from util.laserimage import plotLaserImage


class CalibrationTool(QtWidgets.QDialog):
    def __init__(self, laser, parent=None):
        super().__init__(parent)

        self.laser = laser

        # Left side
        self.lineedit_levels = QtWidgets.QLineEdit()
        self.lineedit_levels.setText("5")
        self.lineedit_levels.setValidator(QtGui.QIntValidator(0, 20))

        self.lineedit_units = QtWidgets.QLineEdit()

        layout_cal_form = QtWidgets.QFormLayout()
        layout_cal_form.addRow("Calibration Levels:", self.lineedit_levels)
        layout_cal_form.addRow("Units:", self.lineedit_units)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Concentration", "Counts"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.table.setRowCount(int(self.lineedit_levels.text()))
        self.table.setVerticalHeaderLabels([c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])

        # Results
        self.lineedit_rsq = QtWidgets.QLineEdit("0.0000")
        self.lineedit_rsq.setReadOnly(True)
        self.lineedit_intercept = QtWidgets.QLineEdit("1.0000")
        self.lineedit_intercept.setReadOnly(True)
        self.lineedit_gradient = QtWidgets.QLineEdit("0.0000")
        self.lineedit_gradient.setReadOnly(True)

        layout_result_form = QtWidgets.QFormLayout()
        layout_result_form.addRow("RSQ:", self.lineedit_rsq)
        layout_result_form.addRow("Intercept:", self.lineedit_intercept)
        layout_result_form.addRow("Gradient:", self.lineedit_gradient)

        box_result = QtWidgets.QGroupBox("Result")
        box_result.setLayout(layout_result_form)

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addLayout(layout_cal_form)
        layout_left.addWidget(self.table)
        layout_left.addWidget(box_result)

        # Right side
        self.button_laser = QtWidgets.QPushButton("Select &Image...")
        self.button_laser.pressed.connect(self.onButtonLaser)

        self.canvas = Canvas(parent=self)

        # Trim
        trim = [0, 0]
        self.lineedit_left = QtWidgets.QLineEdit()
        self.lineedit_left.setPlaceholderText(str(trim[0]))
        self.lineedit_left.setValidator(QtGui.QIntValidator(0, 1e9))
        self.lineedit_right = QtWidgets.QLineEdit()
        self.lineedit_right.setPlaceholderText(str(trim[1]))
        self.lineedit_right.setValidator(QtGui.QIntValidator(0, 1e9))

        self.combo_trim = QtWidgets.QComboBox()
        self.combo_trim.addItems(["rows", "s", "μm"])
        self.combo_trim.setCurrentIndex(1)
        self.combo_trim.currentIndexChanged.connect(self.onComboTrim)

        layout_box_trim = QtWidgets.QHBoxLayout()
        layout_box_trim.addWidget(QtWidgets.QLabel("Left:"))
        layout_box_trim.addWidget(self.lineedit_left)
        layout_box_trim.addWidget(QtWidgets.QLabel("Right:"))
        layout_box_trim.addWidget(self.lineedit_right)
        layout_box_trim.addWidget(self.combo_trim)

        box_trim = QtWidgets.QGroupBox("Trim")
        box_trim.setLayout(layout_box_trim)

        # Bar
        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.onComboIsotope)

        layout_canvas_bar = QtWidgets.QHBoxLayout()
        layout_canvas_bar.addWidget(box_trim)
        layout_canvas_bar.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignTop)

        layout_right = QtWidgets.QVBoxLayout()
        layout_right.addWidget(self.button_laser, 0, QtCore.Qt.AlignRight)
        layout_right.addWidget(self.canvas)
        layout_right.addLayout(layout_canvas_bar)

        # Main
        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_left, 1)
        layout_horz.addLayout(layout_right, 2)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Close
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_horz)
        layout_main.addWidget(buttons)
        self.setLayout(layout_main)

    def draw(self):

        isotope = self.combo_isotope.currentText()
        viewconfig = self.window().viewconfig

        self.image = plotLaserImage(
            self.canvas.fig,
            self.canvas.ax,
            self.laser.get(isotope, calibrated=True, trimmed=True),
            scalebar=False,
            cmap=viewconfig["cmap"],
            interpolation=viewconfig["interpolation"],
            vmin=viewconfig["cmap_range"][0],
            vmax=viewconfig["cmap_range"][1],
            aspect=self.laser.aspect(),
            extent=self.laser.extent(trimmed=True),
        )
        for i in np.linspace(0, 1.0, 7):
            rect = Line2D(
                (0.0, 1.0),
                (i, i),
                transform=self.canvas.ax.transAxes,
                linewidth=2.0,
                color="white",
            )
            self.canvas.ax.add_artist(rect)
            self.canvas.ax.annotate(
                f"{i}",
                (0.0, i),
                xytext=(10, 0),
                textcoords="offset points",
                xycoords="axes fraction",
                horizontalalignment="left",
                verticalalignment="center",
                color="white",
            )

        self.canvas.draw()

    def onComboTrim(self, text):
        pass

    def onComboIsotope(self, text):
        pass

    def onButtonLaser(self):
        pass
