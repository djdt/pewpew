import numpy as np
from PySide2 import QtCore, QtWidgets

from pewpew.graphics.options import GraphicsOptions
from pewpew.graphics.lasergraphicsview import LaserGraphicsView

from typing import List, Tuple


class DrawUndoState(object):
    def __init__(self, pos: Tuple[int, int], data: np.ndarray):
        self.data = data
        self.x1, self.y1 = pos
        self.x2, self.y2 = data.shape + np.array(pos)

    def undo(self, x: np.ndarray) -> None:
        x[self.x1 : self.x2, self.y1 : self.y2] = self.data


class DrawGraphicsView(LaserGraphicsView):
    pass


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    w = QtWidgets.QMainWindow()
    w.statusBar().showMessage("a")
    canvas = DrawGraphicsView(GraphicsOptions())
    w.setCentralWidget(canvas)
    w.show()

    import pew.io

    laser = pew.io.npz.load("/home/tom/Downloads/her 00003.npz")
    canvas.drawData(laser.get("31P"), laser.extent)

    canvas.brush["shape"] = "circle"
    canvas.brush["size"] = 50

    app.exec_()
