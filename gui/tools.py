from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from matplotlib.lines import Line2D

from gui.canvas import Canvas

from util.laser import LaserData
from util.laserimage import plotLaserImage


class CalibrationTool(QtWidgets.QDialog):
    def __init__(self, dockarea, viewconfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Standards Tools")

        self.dockarea = dockarea
        docks = dockarea.visibleDocks()
        self.laser = LaserData() if len(docks) != 1 else docks[0].laser
        self.viewconfig = viewconfig
        self.levels = 5
        self.level_names = [c for c in "ABCDEFGHIJKLMNOPQRST"]

        # Left side
        self.lineedit_levels = QtWidgets.QLineEdit()
        self.lineedit_levels.setText(str(self.levels))
        self.lineedit_levels.setValidator(QtGui.QIntValidator(2, 20))
        self.lineedit_levels.editingFinished.connect(self.onLineEditLevels)

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

        self.canvas = Canvas(connect_mouse_events=False, parent=self)

        # Trim
        self.lineedit_left = QtWidgets.QLineEdit()
        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_left.editingFinished.connect(self.onLineEditTrim)
        self.lineedit_right = QtWidgets.QLineEdit()
        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_right.editingFinished.connect(self.onLineEditTrim)

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

        self.combo_isotope.addItems(self.laser.isotopes())
        self.updateTrim()
        self.updateTableRows()
        self.updateTableLevels()
        self.draw()

    def draw(self):

        isotope = self.combo_isotope.currentText()

        self.canvas.clear()

        self.image = plotLaserImage(
            self.canvas.fig,
            self.canvas.ax,
            self.laser.get(isotope, calibrated=True, trimmed=True),
            scalebar=False,
            cmap=self.viewconfig["cmap"],
            interpolation="none",
            vmin=self.viewconfig["cmaprange"][0],
            vmax=self.viewconfig["cmaprange"][1],
            aspect=self.laser.aspect(),
            extent=self.laser.extent(trimmed=True),
        )

        self.drawLevels()
        self.canvas.draw()

    def drawLevels(self):
        ax_fraction = 1.0 / self.levels
        # Draw lines
        for frac in np.linspace(1.0 - ax_fraction, ax_fraction, self.levels - 1):
            rect = Line2D(
                (-0.1, 1.1),
                (frac, frac),
                transform=self.canvas.ax.transAxes,
                linewidth=2.0,
                color="blue",
            )
            self.canvas.ax.add_artist(rect)
        for i, frac in enumerate(np.linspace(1.0, ax_fraction, self.levels)):
            self.canvas.ax.annotate(
                self.level_names[i],
                (0.0, frac),
                xytext=(self.viewconfig["fontsize"], -self.viewconfig["fontsize"]),
                textcoords="offset points",
                xycoords="axes fraction",
                horizontalalignment="left",
                verticalalignment="center",
                color="white",
            )

    def updateResults(self):
        xs = []
        ys = []

        for level in range(0, self.levels):
            xs.append(float(self.table.item(level, 0).text()))
            ys.append(float(self.table.item(level, 1).text()))
        # np.ployfit
        # np.linalg.lstsq

    def updateTableRows(self):
        self.table.setRowCount(self.levels)
        self.table.setVerticalHeaderLabels(self.level_names)

        for level in range(0, self.levels):
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(level, 1, item)

    def updateTableLevels(self):
        data = self.laser.get(
            self.combo_isotope.currentText(), calibrated=False, trimmed=True
        )
        sections = np.array_split(data, self.levels, axis=0)

        for level in range(0, self.levels):
            mean_conc = np.mean(sections[level])
            self.table.item(level, 1).setText(f"{mean_conc:.4f}")

    def updateTrim(self):
        trim = self.laser.trimAs(self.combo_trim.currentText())
        self.lineedit_left.setText(str(trim[0]))
        self.lineedit_right.setText(str(trim[1]))

    def onLineEditTrim(self):
        if self.lineedit_left.text() == "" or self.lineedit_right.text() == "":
            return
        trim = [float(self.lineedit_left.text()), float(self.lineedit_right.text())]
        self.laser.setTrim(trim, self.combo_trim.currentText())
        self.updateTableLevels()
        self.draw()

    def onLineEditLevels(self):
        if self.lineedit_levels.text() == "":
            return
        self.levels = int(self.lineedit_levels.text())
        self.updateTableRows()
        self.updateTableLevels()
        self.draw()

    def onComboTrim(self, text):
        if self.combo_trim.currentText() == "rows":
            self.lineedit_left.setValidator(QtGui.QIntValidator(0, 1e9))
            self.lineedit_right.setValidator(QtGui.QIntValidator(0, 1e9))
        else:
            self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
            self.lineedit_right.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))

    def onComboIsotope(self, text):
        self.draw()

    def onButtonLaser(self):
        self.hide()
        self.dockarea.activateWindow()
        self.dockarea.setFocus(QtCore.Qt.OtherFocusReason)
        self.dockarea.startMouseSelect()
        self.dockarea.mouseSelectFinished.connect(self.mouseSelectFinished)

    @QtCore.pyqtSlot("QWidget*")
    def mouseSelectFinished(self, widget):
        if widget is not None and hasattr(widget, "laser"):
            self.laser = widget.laser
            self.combo_isotope.clear()
            self.combo_isotope.addItems(self.laser.isotopes())
            self.updateTrim()
            self.updateTableLevels()

        self.dockarea.mouseSelectFinished.disconnect(self.mouseSelectFinished)
        self.activateWindow()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.show()
        self.draw()

    def keyPressEvent(self, event):
        if event.key() in [QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter]:
            return
        super().keyPressEvent(event)
