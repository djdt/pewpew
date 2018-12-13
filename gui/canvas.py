from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(frameon=False, tight_layout=True,
                          figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        return QtCore.QSize(250, 250)

    def clear(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
