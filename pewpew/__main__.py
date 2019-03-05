import sys

from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow

import numpy as np
from fractions import Fraction
from typing import List, Tuple
from pewpew.lib.laser import LaserData
from pewpew.lib.krisskross import krissKrossLayers


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = dict(LaserData.DEFAULT_CONFIG)
    config["spotsize"] = 10.0
    config["speed"] = 10.0
    config["scantime"] = 0.1

    from pewpew.lib.io import agilent

    horz = agilent.load("/home/tom/Downloads/raw/Horz.b", config)
    vert = agilent.load("/home/tom/Downloads/raw/Vert.b", config)

    d = krissKrossLayers(
        [horz.data, vert.data],
        int(horz.aspect()),
        int(12.0 / horz.config["scantime"]),
        [Fraction(1, 2)],
    )[0]["56Fe"]

    plt.imshow(d.mean(axis=2))
    plt.show()


# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)[]

# #     window = MainWindow()
# #     sys.excepthook = window.exceptHook  # type: ignore
# #     window.show()

# #     app.exec()
