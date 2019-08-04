from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from pewpew.ui.canvas.basic import BasicCanvas

from typing import Tuple, Union


class StatsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        data: np.ndarray,
        range: Tuple[Union[str, float], Union[str, float]],
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Statistics")

        self.canvas = BasicCanvas(figsize=(6, 2))
        self.canvas.ax = self.canvas.figure.add_subplot(111)
        # self.canvas.ax.spines["top"].set_visible(False)
        # self.canvas.ax.spines["right"].set_visible(False)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.close)

        stats_left = QtWidgets.QFormLayout()
        stats_left.addRow("Shape:", QtWidgets.QLabel(str(data.shape)))
        stats_left.addRow("Size:", QtWidgets.QLabel(str(data.size)))

        # Ensure no nans
        data = data[~np.isnan(data)]
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

        # Calculate the range
        if isinstance(range[0], str):
            vmin = np.percentile(data, float(range[0].rstrip("%")))
        else:
            vmin = float(range[0])
        if isinstance(range[1], str):
            vmax = np.percentile(data, float(range[1].rstrip("%")))
        else:
            vmax = float(range[1])

        plot_data = data[np.logical_and(data >= vmin, data <= vmax)]
        self.plot(plot_data)

    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        action_copy_image = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy Image", self
        )
        action_copy_image.setStatusTip("Copy image to clipboard.")
        action_copy_image.triggered.connect(self.canvas.copyToClipboard)

        context_menu = QtWidgets.QMenu(self)
        context_menu.addAction(action_copy_image)
        context_menu.popup(event.globalPos())

    def plot(self, data: np.ndarray) -> None:
        highlight = self.palette().color(QtGui.QPalette.Highlight).name()
        self.canvas.ax.hist(data.ravel(), bins="auto", color=highlight)
        self.canvas.draw()
