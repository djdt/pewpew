from PySide2 import QtHelp, QtWidgets
from pathlib import Path


def createHelpWindow(index: str = None):
    print(QtWidgets.QApplication.applicationDirPath())
    path = Path(__file__).parent.joinpath("resources/pewpew.qhc").absolute()
    print(path)
    engine = QtHelp.QHelpEngine(str(path))
    engine.setupData()
