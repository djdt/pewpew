import sys

from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow

import numpy as np
from typing import List, Tuple
from pewpew.lib.laser import LaserData


def subPixelOffset(d: List[np.ndarray], x_offsets: np.ndarray, y_offsets: np.ndarray) -> np.nd.array:
    xgcd = 
    pass

def krissKrossLayers(
    layers: List[np.ndarray],
    offset: Tuple[float, float],
    aspect: float,
    warmup: int,
    horizontal_first: bool = True,
) -> np.ndarray:

    j = 0 if horizontal_first else 1
    # aspect = int(aspect)
    # trim = int(aspect / 2)
    # Calculate the line lengths
    # length = (layers[1].shape[0] * aspect, layers[0].shape[0] * aspect)
    offset = np.array(offset) * 100
    gcd = np.gcd(offset.astype(np.int), 100)
    subpix_offset = 100 // gcd
    pix_size = offset // gcd

    # Reshape the layers and stack into matrix
    transformed = []
    for i, layer in enumerate(layers):
        # Trim data of warmup time and excess
        layer = layer[:, warmup : warmup + length[(i + j) % 2]]
        # Stretch array
        layer = np.repeat(layer, aspect, axis=0)
        # Flip vertical layers and trim
        if (i + j) % 2 == 1:
            layer = layer.T
            layer = layer[trim:, trim:]
        elif trim > 0:
            layer = layer[:-trim, :-trim]

        transformed.append(layer)

    for t in transformed:
        print(t.shape)
    data = np.dstack(transformed)

    return data


if __name__ == "__main__":
    config = dict(LaserData.DEFAULT_CONFIG)
    config["spotsize"] = 30.0
    config["speed"] = 120.0
    config["scantime"] = 0.25

    horz = LaserData(
        np.array(
            [
                [(1), (1), (0), (0), (1), (1), (0), (0), (1), (1)],
                [(0), (0), (1), (1), (0), (0), (1), (1), (0), (0)],
                [(1), (1), (0), (0), (1), (1), (0), (0), (1), (1)],
                [(0), (0), (1), (1), (0), (0), (1), (1), (0), (0)],
                [(1), (1), (0), (0), (1), (1), (0), (0), (1), (1)],
            ],
            dtype=[("a", np.float64)],
        ),
        config,
    )
    vert = LaserData(
        np.array(
            [
                [(0), (0), (0), (0), (0), (0), (0), (0), (0), (0)],
                [(0), (0), (1), (1), (1), (1), (1), (1), (0), (0)],
                [(0), (0), (1), (1), (0), (0), (1), (1), (0), (0)],
                [(0), (0), (1), (1), (1), (1), (1), (1), (0), (0)],
                [(0), (0), (0), (0), (0), (0), (0), (0), (0), (0)],
            ],
            dtype=[("a", np.float64)],
        ),
        config,
    )
    # from pewpew.lib.io import agilent
    # horz = agilent.load("/home/tom/Downloads/raw/Horz.b", config)
    # vert = agilent.load("/home/tom/Downloads/raw/Vert.b", config)

    data = krissKrossLayers([l.data for l in [horz, vert]], 2, 0)

    import matplotlib.pyplot as plt

    plt.imshow(data["a"].mean(axis=2))

    plt.show()


# if __name__ == "__main__":
#     app = QApplication(sys.argv)[]

#     window = MainWindow()
#     sys.excepthook = window.exceptHook  # type: ignore
#     window.show()

#     app.exec()
