import numpy as np
import pytest

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent

from pewpew.lib.mplwidgets import (
    _ImageSelectionWidget,
    LassoImageSelectionWidget,
    RectangleImageSelectionWidget,
)

from typing import Tuple


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


def test_lasso_image_selection_widget():
    fig, ax = plt.subplots()
    img = ax.imshow(np.random.random((50, 50)), extent=(0, 100, 0, 100))
    ax.figure.canvas.draw()

    tool = RectangleImageSelectionWidget(img)
    tool.set_active(True)

    assert not np.any(tool.mask)

    tool.press(FakeEvent(tool.ax, 0, 0))
    tool.onmove(FakeEvent(tool.ax, 50, 50))
    tool.release(FakeEvent(tool.ax, 50, 50))

    assert np.all(tool.mask[25:, :25])
    assert not np.any(tool.mask[:25, 25:])

    tool.press(FakeEvent(tool.ax, 50, 50))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.release(FakeEvent(tool.ax, 75, 75))

    assert not np.any(tool.mask[:12, :25])
    assert np.all(tool.mask[12:25, 25:37])
    assert not np.any(tool.mask[25:, 37:])

    # Test add on shift
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="shift"))
    tool.press(FakeEvent(tool.ax, 0, 0))
    tool.onmove(FakeEvent(tool.ax, 100, 100))
    tool.release(FakeEvent(tool.ax, 100, 100))
    tool.on_key_release(FakeEvent(tool.ax, 100, 100, key="shift"))

    assert np.all(tool.mask)

    # Test subtract on control
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="control"))
    tool.press(FakeEvent(tool.ax, 25, 25))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.release(FakeEvent(tool.ax, 75, 75))
    tool.on_key_release(FakeEvent(tool.ax, 75, 75, key="control"))

    assert np.all(tool.mask[:12, :12])
    assert not np.any(tool.mask[12:37, 12:37])
    assert np.all(tool.mask[37:, 37:])

    # Test clear on escape
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="escape"))
    tool.on_key_release(FakeEvent(tool.ax, 0, 0, key="escape"))
    assert not np.any(tool.mask)
