from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from matplotlib.text import Text

from pewpew.ui.canvas.basic import BasicCanvas

from pewpew.lib.calc import weighted_linreg

from typing import List


class StandardsResultsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray = None,
        unit: str = "",
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.canvas = BasicCanvas(self)
        ax = self.canvas.figure.add_subplot(111)

        m, b, r2 = weighted_linreg(x, y, w)
        x0, x1 = 0.0, x.max() * 1.1

        xlabel = "Concentration"
        if unit != "":
            xlabel += f" ({unit})"

        ax.scatter(x, y, color="black")
        ax.plot([x0, x1], [m * x0 + b, m * x1 + b], ls=":", lw=1.5, color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")

        text = Text(
            x=0.05,
            y=0.95,
            text=f"y = {m:.4f} · x - {b:.4f}\nr² = {r2:.4f}",
            transform=ax.transAxes,
            color="black",
            fontsize=12,
            horizontalalignment="left",
            verticalalignment="top",
        )

        ax.add_artist(text)

        self.canvas.draw()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        context_menu = QtWidgets.QMenu(self)
        action_copy = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy Image", self
        )
        action_copy.setStatusTip("Copy image to clipboard.")
        action_copy.triggered.connect(self.copyImage)
        context_menu.addAction(action_copy)
        context_menu.exec(event.globalPos())

    def copyImage(self) -> None:
        QtWidgets.QApplication.clipboard().setPixmap(self.canvas.grab())


class StandardsResultsBox(QtWidgets.QGroupBox):
    LABELS = ["RSQ", "Gradient", "Intercept"]

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__("Results", parent)
        self.lineedits: List[QtWidgets.QLineEdit] = []
        self.button = QtWidgets.QPushButton("Plot")
        self.button.setEnabled(False)

        self.r2 = 0.0
        self.m = 0.0
        self.b = 0.0

        layout = QtWidgets.QFormLayout()

        for label in StandardsResultsBox.LABELS:
            le = QtWidgets.QLineEdit()
            le.setReadOnly(True)

            layout.addRow(label, le)
            self.lineedits.append(le)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.button, 0, QtCore.Qt.AlignRight)
        layout.addRow(button_layout)
        self.setLayout(layout)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        copy_action = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("edit-copy"), "Copy All", self
        )
        copy_action.triggered.connect(self.copy)

        menu.addAction(copy_action)

        menu.exec(event.globalPos())

    def copy(self) -> None:
        data = (
            '<meta http-equiv="content-type" content="text/html; charset=utf-8"/>'
            "<table>"
        )
        text = ""

        for label, lineedit in zip(StandardsResultsBox.LABELS, self.lineedits):
            value = lineedit.text()
            data += f"<tr><td>{label}</td><td>{value}</td></tr>"
            text += f"{label}\t{value}\n"
        data += "</table>"

        mime = QtCore.QMimeData()
        mime.setHtml(data)
        mime.setText(text)
        QtWidgets.QApplication.clipboard().setMimeData(mime)

    def clear(self) -> None:
        for le in self.lineedits:
            le.setText("")
        self.button.setEnabled(False)

    def update(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        self.m, self.b, self.r2 = weighted_linreg(x, y, w=w)
        for v, le in zip([self.r2, self.m, self.b], self.lineedits):
            le.setText(f"{v:.4f}")
        self.button.setEnabled(True)
