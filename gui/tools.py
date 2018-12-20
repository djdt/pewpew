from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gui.canvas import Canvas
from gui.widgets import CalibrationTable

from util.laser import LaserData
from util.laserimage import plotLaserImage


class CalibrationTool(QtWidgets.QDialog):
    def __init__(self, dockarea, viewconfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Standards Tool")

        self.dockarea = dockarea
        docks = dockarea.orderedDocks(dockarea.visibleDocks())
        self.laser = LaserData() if len(docks) < 1 else docks[0].laser
        self.viewconfig = viewconfig
        self.previous_isotope = None
        self.concentrations = {}

        # Left side
        self.spinbox_levels = QtWidgets.QSpinBox()

        self.table = CalibrationTable(self)

        self.lineedit_units = QtWidgets.QLineEdit()

        self.lineedit_gradient = QtWidgets.QLineEdit()
        self.lineedit_intercept = QtWidgets.QLineEdit()
        self.lineedit_rsq = QtWidgets.QLineEdit()

        # Right side
        self.button_laser = QtWidgets.QPushButton("Select &Image...")

        self.canvas = Canvas(connect_mouse_events=False, parent=self)

        self.lineedit_left = QtWidgets.QLineEdit()
        self.lineedit_right = QtWidgets.QLineEdit()

        self.combo_trim = QtWidgets.QComboBox()

        self.combo_isotope = QtWidgets.QComboBox()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply,
            self,
        )

        self.initialiseWidgets()
        self.layoutWidgets()

        self.previous_isotope = self.combo_isotope.currentText()

        self.updateTrim()
        self.table.updateCounts(
            self.laser.get(
                self.combo_isotope.currentText(), calibrated=False, trimmed=True
            )
        )
        self.draw()

    def initialiseWidgets(self):
        self.spinbox_levels.setMaximum(20)
        self.spinbox_levels.setValue(5)
        self.spinbox_levels.valueChanged.connect(self.onSpinBoxLevels)

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_weighting.addItems(["x", "1/x", "1/(x^2)"])
        self.combo_weighting.currentIndexChanged.connect(self.onComboWeighting)

        self.table.itemChanged.connect(self.onTableItemChanged)
        self.table.setRowCount(5)

        self.lineedit_gradient.setReadOnly(True)
        self.lineedit_intercept.setReadOnly(True)
        self.lineedit_rsq.setReadOnly(True)

        self.button_laser.pressed.connect(self.onButtonLaser)

        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_left.editingFinished.connect(self.onLineEditTrim)
        self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.lineedit_right.editingFinished.connect(self.onLineEditTrim)

        self.combo_trim.addItems(["rows", "s", "μm"])
        self.combo_trim.setCurrentIndex(1)
        self.combo_trim.currentIndexChanged.connect(self.onComboTrim)

        self.combo_isotope.addItems(self.laser.isotopes())
        self.combo_isotope.currentIndexChanged.connect(self.onComboIsotope)

        self.button_box.clicked.connect(self.buttonBoxClicked)

    def layoutWidgets(self):
        layout_cal_form = QtWidgets.QFormLayout()
        layout_cal_form.addRow("Calibration Levels:", self.spinbox_levels)

        layout_table_form = QtWidgets.QFormLayout()
        layout_table_form.addRow("Units:", self.lineedit_units)
        layout_table_form.addRow("Weighting:", self.combo_weighting)

        layout_result_form = QtWidgets.QFormLayout()
        layout_result_form.addRow("RSQ:", self.lineedit_rsq)
        layout_result_form.addRow("Intercept:", self.lineedit_intercept)
        layout_result_form.addRow("Gradient:", self.lineedit_gradient)

        box_result = QtWidgets.QGroupBox("Result")
        box_result.setLayout(layout_result_form)

        layout_left = QtWidgets.QVBoxLayout()
        layout_left.addLayout(layout_cal_form)
        layout_left.addWidget(self.table)
        layout_left.addLayout(layout_table_form)
        layout_left.addWidget(box_result)

        layout_box_trim = QtWidgets.QHBoxLayout()
        layout_box_trim.addWidget(QtWidgets.QLabel("Left:"))
        layout_box_trim.addWidget(self.lineedit_left)
        layout_box_trim.addWidget(QtWidgets.QLabel("Right:"))
        layout_box_trim.addWidget(self.lineedit_right)
        layout_box_trim.addWidget(self.combo_trim)

        box_trim = QtWidgets.QGroupBox("Trim")
        box_trim.setLayout(layout_box_trim)

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

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_horz)
        layout_main.addWidget(self.button_box)
        self.setLayout(layout_main)

    def draw(self):
        self.canvas.clear()

        isotope = self.combo_isotope.currentText()
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
        levels = self.spinbox_levels.value()
        ax_fraction = 1.0 / levels
        # Draw lines
        for frac in np.linspace(1.0 - ax_fraction, ax_fraction, levels - 1):
            line = Line2D(
                (0.0, 1.0),
                (frac, frac),
                transform=self.canvas.ax.transAxes,
                color="black",
                linestyle="--",
                linewidth=2.0,
            )
            self.canvas.ax.add_artist(line)

        # Bookend for text
        div = make_axes_locatable(self.canvas.ax)
        cax = div.append_axes('left', size=0.2, pad=0, sharey=self.canvas.ax)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.set_facecolor("black")

        for i, frac in enumerate(np.linspace(1.0, ax_fraction, levels)):
            text = Text(
                x=0.5,
                y=frac - (ax_fraction / 2.0),
                text=CalibrationTable.ROW_LABELS[i],
                transform=cax.transAxes,
                color="white",
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="center",
            )
            cax.add_artist(text)

    def updateResults(self):
        if not self.table.complete():
            return
        m, b, r2 = self.table.calibrationResults(
            self.combo_weighting.currentText())
        self.lineedit_gradient.setText(f"{m:.4f}")
        self.lineedit_intercept.setText(f"{b:.4f}")
        self.lineedit_rsq.setText(f"{r2:.4f}")

    def updateTrim(self):
        trim = self.laser.trimAs(self.combo_trim.currentText())
        self.lineedit_left.setText(str(trim[0]))
        self.lineedit_right.setText(str(trim[1]))

    def buttonBoxClicked(self, button):
        sb = self.button_box.standardButton(button)

        if sb == QtWidgets.QDialogButtonBox.Apply:
            self.apply()
        elif sb == QtWidgets.QDialogButtonBox.Ok:
            self.accept()
        else:
            self.reject()

    def onButtonLaser(self):
        self.hide()
        self.dockarea.activateWindow()
        self.dockarea.setFocus(QtCore.Qt.OtherFocusReason)
        self.dockarea.startMouseSelect()
        self.dockarea.mouseSelectFinished.connect(self.mouseSelectFinished)

    def onComboTrim(self, text):
        if self.combo_trim.currentText() == "rows":
            self.lineedit_left.setValidator(QtGui.QIntValidator(0, 1e9))
            self.lineedit_right.setValidator(QtGui.QIntValidator(0, 1e9))
        else:
            self.lineedit_left.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
            self.lineedit_right.setValidator(QtGui.QDoubleValidator(0, 1e9, 2))
        self.updateTrim()

    def onComboIsotope(self, text):
        isotope = self.combo_isotope.currentText()
        self.concentrations[self.previous_isotope] = self.table.concentrations()

        self.table.updateConcentrations(self.concentrations.get(isotope, None))
        self.table.updateCounts(self.laser.get(
            isotope, calibrated=False, trimmed=True))
        self.updateResults()
        self.draw()
        self.previous_isotope = isotope

    def onComboWeighting(self, text):
        self.updateResults()

    def onLineEditTrim(self):
        if self.lineedit_left.text() == "" or self.lineedit_right.text() == "":
            return
        trim = [float(self.lineedit_left.text()),
                float(self.lineedit_right.text())]
        self.laser.setTrim(trim, self.combo_trim.currentText())
        self.table.updateCounts(
            self.laser.get(
                self.combo_isotope.currentText(), calibrated=False, trimmed=True
            )
        )
        self.updateResults()
        self.draw()

    def onSpinBoxLevels(self):
        self.table.setRowCount(self.spinbox_levels.value())
        self.table.updateCounts(
            self.laser.get(
                self.combo_isotope.currentText(), calibrated=False, trimmed=True
            )
        )
        self.updateResults()
        self.draw()

    def onTableItemChanged(self, item):
        if item.text() != "":
            self.updateResults()

    @QtCore.pyqtSlot("QWidget*")
    def mouseSelectFinished(self, widget):
        if widget is not None and hasattr(widget, "laser"):
            self.laser = widget.laser
            # Prevent currentIndexChanged being emmited
            self.combo_isotope.blockSignals(True)
            self.combo_isotope.clear()
            self.combo_isotope.addItems(self.laser.isotopes())
            self.combo_isotope.blockSignals(False)

            self.updateTrim()
            self.table.updateCounts(
                self.laser.get(
                    self.combo_isotope.currentText(), calibrated=False, trimmed=True
                )
            )
            self.updateResults()

        self.dockarea.mouseSelectFinished.disconnect(self.mouseSelectFinished)
        self.activateWindow()
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self.show()
        self.draw()

    def keyPressEvent(self, event):
        if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
            return
        super().keyPressEvent(event)

    def accept(self):
        pass

    def apply(self):
        pass
