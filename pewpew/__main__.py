import sys

from PyQt5.QtWidgets import QApplication
from pewpew.ui import MainWindow

import numpy as np
from typing import List, Tuple
from pewpew.lib.laser import LaserData


def subpixelOffset(
    images: List[np.ndarray], offsets: List[Tuple[int, int]], pixelsize: Tuple[int, int]
) -> np.ndarray:
    if offsets[0] != (0, 0):  # The zero offset
        offsets.insert(0, (0, 0))
    overlap = np.max(offsets, axis=0)
    shape = images[0].shape
    dtype = images[0].dtype

    for img in images:
        if img.ndim != 2:
            raise ValueError("Array must be 2 dimensional.")
        if img.shape != shape:
            raise ValueError("Arrays must have same shape.")
        if img.dtype != dtype:
            raise ValueError("Arrays must have same dtype.")

    new_shape = np.array(shape) * pixelsize + overlap
    data = np.zeros((*new_shape, len(images)), dtype=dtype)
    for i, img in enumerate(images):
        start = offsets[i % len(offsets)]
        end = -(overlap[0] - start[0]) or None, -(overlap[1] - start[1]) or None
        data[start[0] : end[0], start[1] : end[1], i] = np.repeat(
            img, pixelsize[0], axis=0
        ).repeat(pixelsize[1], axis=1)

    return data


# def subpixelOffset(
#     images: List[np.ndarray], offsets: List[int], pixelsize: int
# ) -> np.ndarray:
#     if offsets[0] != 0:  # The zero offset
#         offsets.insert(0, 0)
#     overlap = np.max(offsets)
#     shape = images[0].shape
#     dtype = images[0].dtype

#     for img in images:
#         if img.ndim != 2:
#             raise ValueError("Array must be 2 dimensional.")
#         if img.shape != shape:
#             raise ValueError("Arrays must have same shape.")
#         if img.dtype != dtype:
#             raise ValueError("Arrays must have same dtype.")

#     data = np.zeros(
#         (shape[0] * pixelsize + overlap, shape[1] * pixelsize + overlap, len(images)),
#         dtype=dtype,
#     )
#     for i, img in enumerate(images):
#         start = offsets[i % len(offsets)]
#         end = -(overlap - start) or None
#         data[start:end, start:end, i] = np.repeat(img, pixelsize, axis=0).repeat(
#             pixelsize, axis=1
#         )

#     return data


# def subpixelEqualOffset(images: List[np.ndarray]) -> np.ndarray:
# return subpixelOffset(images, np.arange(0, len(images), 1))


def krissKrossLayers(
    layers: List[np.ndarray], aspect: float, warmup: int, horizontal_first: bool = True
) -> np.ndarray:

    j = 0 if horizontal_first else 1
    aspect = int(aspect)
    trim = int(aspect / 2)
    # Calculate the line lengths
    length = (layers[1].shape[0] * aspect, layers[0].shape[0] * aspect)

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

    return np.dstack(transformed)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = dict(LaserData.DEFAULT_CONFIG)
    config["spotsize"] = 10.0
    config["speed"] = 10.0
    config["scantime"] = 0.1

    h = np.array(
        [
            [(1), (1), (1), (1), (1)],
            [(0), (0), (0), (0), (0)],
            [(1), (1), (1), (1), (1)],
            [(0), (0), (0), (0), (0)],
            [(1), (1), (1), (1), (1)],
        ],
        dtype=np.float64,
    )
    v = np.array(
        [
            [(1), (0), (1), (0), (1)],
            [(1), (0), (1), (0), (1)],
            [(1), (0), (1), (0), (1)],
            [(1), (0), (1), (0), (1)],
            [(1), (0), (1), (0), (1)],
        ],
        dtype=np.float64,
    )

    # d = subpixelEqualOffset([h, v])
    d = subpixelOffsetXY([h, v, h, v, h, v, h, v, h, v], [(1, 2), (2, 3)], (1, 1))
    plt.imshow(d.mean(axis=2))
    plt.show()
    from pewpew.lib.io import agilent

    # horz = agilent.load("/home/tom/Downloads/raw/Horz.b", config)
    # vert = agilent.load("/home/tom/Downloads/raw/Vert.b", config)


    # data = krissKrossLayers([l.data for l in [horz, vert]], 2, 0)

    # import matplotlib.pyplot as plt

    # plt.imshow(data["a"].mean(axis=2))

    # plt.show()


# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)[]

# #     window = MainWindow()
# #     sys.excepthook = window.exceptHook  # type: ignore
# #     window.show()

# #     app.exec()
