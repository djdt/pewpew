import sys

from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow

from pewpew.lib.io import npz
from pewpew.ui.docks import LaserImageDock


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    sys.excepthook = window.exceptHook  # type: ignore
    window.show()

    ld = npz.load("/home/tom/Dropbox/Uni/Experimental/LAICPMS/agilent/20190123_mn_tb2911.npz")
    window.dockarea.addDockWidgets([LaserImageDock(ld[0], window.dockarea)])

    app.exec()
