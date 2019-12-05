import numpy as np

from matplotlib.axes import Axes

from typing import List, Union


def linear_data(names: Union[str, List[str]]) -> np.ndarray:
    names = names if isinstance(names, list) else [names]
    dtype = [(name, float) for name in names]
    data = np.empty((10, 10), dtype=dtype)
    for name in names:
        data[name] = np.repeat(
            np.arange(0, 10, dtype=float).reshape((10, 1)), 10, axis=1
        )
    return data


def rand_data(names: Union[str, List[str]]) -> np.ndarray:
    names = names if isinstance(names, list) else [names]
    dtype = [(name, float) for name in names]
    data = np.empty((10, 10), dtype=dtype)
    for name in names:
        data[name] = np.random.random((10, 10))
    return data


class FakeEvent(object):
    def __init__(
        self, ax: Axes, xdata: float, ydata: float, key: str = None, button: int = 1
    ):
        self.inaxes = ax
        self.canvas = ax.figure.canvas
        self.xdata, self.ydata = xdata, ydata
        self.x, self.y = ax.transData.transform((xdata, ydata))
        self.key = key
        self.button = button

        self.guiEvent = None
        self.name = "none"
