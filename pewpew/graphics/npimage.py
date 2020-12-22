import numpy as np
from PySide2 import QtGui


def array_to_image(array: np.ndarray, vmin: float = None, vmax: float = None):
    if vmin is None:
        vmin = np.amin(array)
    if vmax is None:
        vmax = np.amax(array)
    if array.ndim != 2:
        raise ValueError("Array must be 2 dimensional.")

    data = np.clip(array, vmin, vmax)
    data = (data - vmin) / (vmax - vmin)
    data = data * 255.0
    data = data.astype(np.uint8)

    return QtGui.QImage(
        data.data,
        data.shape[1],
        data.shape[0],
        data.strides[0],
        QtGui.QImage.Format_Indexed8,
    )


def float_image_to_indexed8(array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    data = np.clip(array, vmin, vmax)
    data = (data - vmin) / (vmax - vmin)
    data = data * 255.0
    return data.astype(np.uint8)


class NumpyImage(QtGui.QImage):
    def __init__(self, array: np.ndarray, vmin: float = None, vmax: float = None):
        if array.ndim != 2:
            raise ValueError("Array must be 2 dimensional.")

        if vmin is None:
            vmin = np.amin(array)
        if vmax is None:
            vmax = np.amax(array)
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        self.vmin, self.vmax = vmin, vmax
        self.array = array
        self.data = float_image_to_indexed8(self.array, self.vmin, self.vmax)

        super().__init__(
            self.data.data,
            self.data.shape[1],
            self.data.shape[0],
            self.data.strides[0],
            QtGui.QImage.Format_Indexed8,
        )
