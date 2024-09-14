from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np

from pewpew.charts.base import SinglePlotGraphicsView


class SignalView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
