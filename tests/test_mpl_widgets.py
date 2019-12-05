import numpy as np

import matplotlib.pyplot as plt

from pewpew.lib.mplwidgets import (
    LassoImageSelectionWidget,
    RectangleImageSelectionWidget,
)

from testing import FakeEvent

from typing import Set

mask = np.zeros((50, 50), dtype=bool)


def update_mask(m: np.ndarray, s: Set) -> None:
    global mask
    if "add" in s:
        mask = np.logical_or(mask, m)
    elif "subtract" in s:
        mask = np.logical_and(mask, ~m)
    else:
        mask = m


def test_lasso_image_selection_widget():
    global mask

    fig, ax = plt.subplots()
    img = ax.imshow(np.random.random((50, 50)), extent=(0, 100, 0, 100))
    ax.figure.canvas.draw()

    tool = LassoImageSelectionWidget(img, update_mask)
    tool.set_active(True)

    assert not np.any(mask)

    tool.press(FakeEvent(tool.ax, 50, 50))
    tool.onmove(FakeEvent(tool.ax, 50, 75))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.onmove(FakeEvent(tool.ax, 75, 50))
    tool.release(FakeEvent(tool.ax, 75, 50))

    assert not np.any(mask[:12, :25])
    assert np.all(mask[12:25, 25:37])
    assert not np.any(mask[25:, 37:])

    tool.press(FakeEvent(tool.ax, 0, 0))
    tool.onmove(FakeEvent(tool.ax, 0, 50))
    tool.onmove(FakeEvent(tool.ax, 100, 50))
    tool.onmove(FakeEvent(tool.ax, 100, 0))
    tool.release(FakeEvent(tool.ax, 100, 0))

    assert np.all(mask[25:, :25])
    assert not np.any(mask[:25, 25:])

    # Test add on shift
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="shift"))
    tool.press(FakeEvent(tool.ax, 100, 50))
    tool.onmove(FakeEvent(tool.ax, 100, 100))
    tool.onmove(FakeEvent(tool.ax, 0, 100))
    tool.onmove(FakeEvent(tool.ax, 0, 50))
    tool.release(FakeEvent(tool.ax, 0, 50))
    tool.on_key_release(FakeEvent(tool.ax, 0, 50, key="shift"))

    assert np.all(mask)

    # Test subtract on control
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="control"))
    tool.press(FakeEvent(tool.ax, 24, 25))  # 24?
    tool.onmove(FakeEvent(tool.ax, 75, 25))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.onmove(FakeEvent(tool.ax, 25, 75))
    tool.release(FakeEvent(tool.ax, 25, 75))
    tool.on_key_release(FakeEvent(tool.ax, 25, 75, key="control"))

    assert np.all(mask[:12, :12])
    assert not np.any(mask[12:37, 12:37])
    assert np.all(mask[37:, 37:])

    # Test clear on escape
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="escape"))
    tool.on_key_release(FakeEvent(tool.ax, 0, 0, key="escape"))
    assert not np.any(mask)


def test_rectangle_image_selection_widget():
    global mask
    mask[:] = False

    fig, ax = plt.subplots()
    img = ax.imshow(np.random.random((50, 50)), extent=(0, 100, 0, 100))
    ax.figure.canvas.draw()

    tool = RectangleImageSelectionWidget(img, update_mask)
    tool.set_active(True)

    assert not np.any(mask)

    tool.press(FakeEvent(tool.ax, 50, 50))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.release(FakeEvent(tool.ax, 75, 75))

    assert not np.any(mask[:12, :25])
    assert np.all(mask[12:25, 25:37])
    assert not np.any(mask[25:, 37:])

    tool.press(FakeEvent(tool.ax, 0, 0))
    tool.onmove(FakeEvent(tool.ax, 100, 50))
    tool.release(FakeEvent(tool.ax, 100, 50))

    assert np.all(mask[25:, :25])
    assert not np.any(mask[:25, 25:])

    # Test add on shift
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="shift"))
    tool.press(FakeEvent(tool.ax, 100, 50))
    tool.onmove(FakeEvent(tool.ax, 0, 100))
    tool.release(FakeEvent(tool.ax, 0, 100))
    tool.on_key_release(FakeEvent(tool.ax, 0, 100, key="shift"))

    assert np.all(mask)

    # Test subtract on control
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="control"))
    tool.press(FakeEvent(tool.ax, 25, 25))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.release(FakeEvent(tool.ax, 75, 75))
    tool.on_key_release(FakeEvent(tool.ax, 75, 75, key="control"))

    assert np.all(mask[:12, :12])
    assert not np.any(mask[12:37, 12:37])
    assert np.all(mask[37:, 37:])

    # Test clear on escape
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="escape"))
    tool.on_key_release(FakeEvent(tool.ax, 0, 0, key="escape"))
    assert not np.any(mask)
