import sys

from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow


import numpy as np
from typing import List, Tuple
from pewpew.lib.laser import LaserData
from pewpew.lib.krisskross import krissKrossLayers


if __name__ == "__main__":
    config = dict(LaserData.DEFAULT_CONFIG)
    config["spotsize"] = 10.0
    config["speed"] = 10.0
    config["scantime"] = 0.1

    horz = LaserData(
        np.array(
            [
                [(1), (0), (1), (0), (1)],
                [(0), (1), (0), (1), (0)],
                [(1), (0), (1), (0), (1)],
                [(0), (1), (0), (1), (0)],
                [(1), (0), (1), (0), (1)],
            ],
            dtype=[("a", np.float64)],
        ),
        config,
    )
    vert = LaserData(
        np.array(
            [
                [(0), (0), (0), (0), (0)],
                [(0), (1), (1), (1), (0)],
                [(0), (1), (0), (1), (0)],
                [(0), (1), (1), (1), (0)],
                [(0), (0), (0), (0), (0)],
            ],
            dtype=[("a", np.float64)],
        ),
        config,
    )
    # from pewpew.lib.io import agilent
    # horz = agilent.load("/home/tom/Downloads/raw/Horz.b", config)
    # vert = agilent.load("/home/tom/Downloads/raw/Vert.b", config)

    data = krissKrossLayers(
        [l.data for l in [horz, vert]],
        1, 0
    )

    import matplotlib.pyplot as plt

    plt.imshow(data["a"].mean(axis=2))

    plt.show()


# if __name__ == "__main__":
#     app = QApplication(sys.argv)[]

#     window = MainWindow()
#     sys.excepthook = window.exceptHook  # type: ignore
#     window.show()

#     app.exec()
