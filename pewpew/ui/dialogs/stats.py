from PyQt5 import QtGui, QtWidgets
import numpy as np

from pewpew.ui.canvas.basic import BasicCanvas


class StatsDialog(QtWidgets.QDialog):
    def __init__(self, data: np.ndarray, parent: QtWidgets.QWidget = None):
        self.data = data
        super().__init__(parent)

        self.canvas = BasicCanvas(figsize=(6, 2))
        self.canvas.ax = self.canvas.figure.subplots()

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.close)

        stats_left = QtWidgets.QFormLayout()
        stats_left.addRow("Shape:", QtWidgets.QLabel(str(data.shape)))
        stats_left.addRow("Size:", QtWidgets.QLabel(str(data.size)))

        stats_right = QtWidgets.QFormLayout()
        stats_right.addRow("Min:", QtWidgets.QLabel(f"{np.min(data):.4g}"))
        stats_right.addRow("Max:", QtWidgets.QLabel(f"{np.max(data):.4g}"))
        stats_right.addRow("Mean:", QtWidgets.QLabel(f"{np.mean(data):.4g}"))
        stats_right.addRow("Median:", QtWidgets.QLabel(f"{np.median(data):.4g}"))

        stats_box = QtWidgets.QGroupBox()
        stats_layout = QtWidgets.QHBoxLayout()
        stats_layout.addLayout(stats_left)
        stats_layout.addLayout(stats_right)
        stats_box.setLayout(stats_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(stats_box)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.plot(data)

    def plot(self, data: np.ndarray) -> None:
        highlight = self.palette().color(QtGui.QPalette.Highlight).name()
        self.canvas.ax.hist(data.ravel(), bins='auto', color=highlight)
        self.canvas.draw()
