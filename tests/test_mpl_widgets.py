import numpy as np

import matplotlib.pyplot as plt

from pewpew.lib.mplwidgets import (
    LassoImageSelectionWidget,
    RectangleImageSelectionWidget,
    RulerWidget,
)

from testing import FakeEvent

from typing import Set


class SelectionTestObj(object):
    def __init__(self) -> None:
        self.mask = np.zeros((50, 50), dtype=bool)

    def update_mask(self, m: np.ndarray, s: Set) -> None:
        if "add" in s:
            self.mask = np.logical_or(self.mask, m)
        elif "subtract" in s:
            self.mask = np.logical_and(self.mask, ~m)
        else:
            self.mask = m


def test_lasso_image_selection_widget():
    testobj = SelectionTestObj()

    fig, ax = plt.subplots()
    img = ax.imshow(np.random.random((40, 40)), extent=(0, 100, 0, 100))
    ax.figure.canvas.draw()

    tool = LassoImageSelectionWidget(img, testobj.update_mask)
    tool.set_active(True)

    assert not np.any(testobj.mask)

    tool.press(FakeEvent(tool.ax, 50, 50))
    tool.onmove(FakeEvent(tool.ax, 50, 75))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.onmove(FakeEvent(tool.ax, 75, 50))
    tool.release(FakeEvent(tool.ax, 75, 50))

    assert not np.any(testobj.mask[:10, :20])
    assert np.all(testobj.mask[10:20, 20:30])
    assert not np.any(testobj.mask[20:, 30:])

    tool.press(FakeEvent(tool.ax, 0, 0))
    tool.onmove(FakeEvent(tool.ax, 0, 50))
    tool.onmove(FakeEvent(tool.ax, 100, 50))
    tool.onmove(FakeEvent(tool.ax, 100, 0))
    tool.release(FakeEvent(tool.ax, 100, 0))

    assert np.all(testobj.mask[20:, :20])
    assert not np.any(testobj.mask[:20, 20:])

    # Test add on shift
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="shift"))
    tool.press(FakeEvent(tool.ax, 100, 50))
    tool.onmove(FakeEvent(tool.ax, 100, 100))
    tool.onmove(FakeEvent(tool.ax, 0, 100))
    tool.onmove(FakeEvent(tool.ax, 0, 50))
    tool.release(FakeEvent(tool.ax, 0, 50))
    tool.on_key_release(FakeEvent(tool.ax, 0, 50, key="shift"))

    assert np.all(testobj.mask)

    # Test subtract on control
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="control"))
    tool.press(FakeEvent(tool.ax, 24, 25))  # 24?
    tool.onmove(FakeEvent(tool.ax, 75, 25))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.onmove(FakeEvent(tool.ax, 25, 75))
    tool.release(FakeEvent(tool.ax, 25, 75))
    tool.on_key_release(FakeEvent(tool.ax, 25, 75, key="control"))

    assert np.all(testobj.mask[:10, :10])
    assert not np.any(testobj.mask[10:30, 10:30])
    assert np.all(testobj.mask[30:, 30:])

    # Test clear on escape
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="escape"))
    tool.on_key_release(FakeEvent(tool.ax, 0, 0, key="escape"))
    assert not np.any(testobj.mask)


def test_rectangle_image_selection_widget():
    testobj = SelectionTestObj()

    fig, ax = plt.subplots()
    img = ax.imshow(np.random.random((40, 40)), extent=(0, 100, 0, 100))
    ax.figure.canvas.draw()

    tool = RectangleImageSelectionWidget(img, testobj.update_mask)
    tool.set_active(True)

    assert not np.any(testobj.mask)

    tool.press(FakeEvent(tool.ax, 50, 50))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.release(FakeEvent(tool.ax, 75, 75))

    assert not np.any(testobj.mask[:10, :20])
    assert np.all(testobj.mask[10:20, 20:30])
    assert not np.any(testobj.mask[20:, 30:])

    tool.press(FakeEvent(tool.ax, 0, 0))
    tool.onmove(FakeEvent(tool.ax, 100, 50))
    tool.release(FakeEvent(tool.ax, 100, 50))

    assert np.all(testobj.mask[20:, :20])
    assert not np.any(testobj.mask[:20, 20:])

    # Test add on shift
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="shift"))
    tool.press(FakeEvent(tool.ax, 100, 50))
    tool.onmove(FakeEvent(tool.ax, 0, 100))
    tool.release(FakeEvent(tool.ax, 0, 100))
    tool.on_key_release(FakeEvent(tool.ax, 0, 100, key="shift"))

    assert np.all(testobj.mask)

    # Test subtract on control
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="control"))
    tool.press(FakeEvent(tool.ax, 25, 25))
    tool.onmove(FakeEvent(tool.ax, 75, 75))
    tool.release(FakeEvent(tool.ax, 75, 75))
    tool.on_key_release(FakeEvent(tool.ax, 75, 75, key="control"))

    assert np.all(testobj.mask[:10, :10])
    assert not np.any(testobj.mask[10:30, 10:30])
    assert np.all(testobj.mask[30:, 30:])

    # Test clear on escape
    tool.on_key_press(FakeEvent(tool.ax, 0, 0, key="escape"))
    tool.on_key_release(FakeEvent(tool.ax, 0, 0, key="escape"))
    assert not np.any(testobj.mask)


def test_ruler_widget():
    class TestObj(object):
        def set_distance(self, d: float) -> None:
            self.d = d

    testobj = TestObj()
    fig, ax = plt.subplots()
    ax.imshow(np.random.random((50, 50)), extent=(0, 100, 0, 100))
    ax.figure.canvas.draw()

    tool = RulerWidget(ax, testobj.set_distance, drawtext=True)
    tool.set_active(True)

    tool.press(FakeEvent(tool.ax, 0, 0))
    tool.onmove(FakeEvent(tool.ax, 100, 50))

    assert tool.line.get_visible()
    assert tool.text.get_visible()
    assert tool.text.get_text() == f"{np.sqrt(100 ** 2 + 50 ** 2):.2f}"

    tool.release(FakeEvent(tool.ax, 100, 50))

    assert not tool.line.get_visible()
    assert not tool.text.get_visible()
    assert testobj.d == np.sqrt(100 ** 2 + 50 ** 2)
