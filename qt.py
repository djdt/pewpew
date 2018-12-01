# import sys

# from PyQt5.QtWidgets import QApplication
# from gui.qt.mainwindow import MainWindow

from util.importer import importAgilentBatch
from util.exporter import exportVtr
from util.krisskross import KrissKrossData

if __name__ == "__main__":
    # app = QApplication(sys.argv)

    config = {'spotsize': 10.0, 'speed': 10.0, 'scantime': 0.1,
              'gradient': 1.0, 'intercept': 0.0}
    lds = [importAgilentBatch("/home/tom/Downloads/raw/Horz.b", config),
           importAgilentBatch("/home/tom/Downloads/raw/Vert.b", config)]
    kd = KrissKrossData(None, config=config)
    kd.fromLayers([ld.data for ld in lds])

    exportVtr("/home/tom/Downloads/a.vtr",
              kd.calibrated(), kd.extent(),
              kd.config['spotsize'])

    # window = MainWindow()
    # sys.excepthook = window.exceptHook
    # window.show()

    # app.exec()
