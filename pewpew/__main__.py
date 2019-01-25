import sys

from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow
from pewpew.lib.laser import LaserData


if __name__ == "__main__":
    # app = QApplication(sys.argv)

    # window = MainWindow()
    # sys.excepthook = window.exceptHook  # type: ignore
    # window.show()

    # app.exec()

    from pewpew.lib.io import agilent, vtk

    ld = agilent.load(
        "/home/tom/Downloads/20190123_mn_tb2911.b", LaserData.DEFAULT_CONFIG
    )
    vtk.save("/home/tom/Downloads/out.vtr", ld)
