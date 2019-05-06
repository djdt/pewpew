from PyQt5 import QtCore, QtWidgets

from matplotlib.backend_bases import MouseEvent

from typing import Callable, List
from matplotlib.axes import Axes


class DragSelector(QtWidgets.QRubberBand):
    def __init__(
        self,
        button: int = 1,
        callback: Callable = None,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(QtWidgets.QRubberBand.Rectangle, parent)
        self.button = button
        self.callback = callback
        self.extent = (0, 0, 0, 0)
        self.origin = QtCore.QPoint()
        self.cids: List[int] = []

    def _press(self, event: MouseEvent) -> None:
        self.event_press = event
        if event.button != self.button:
            return
        self.origin = event.guiEvent.pos()
        self.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
        self.show()

    def _move(self, event: MouseEvent) -> None:
        if event.button != self.button:
            return
        self.setGeometry(QtCore.QRect(self.origin, event.guiEvent.pos()).normalized())

    def _release(self, event: MouseEvent) -> None:
        self.event_release = event
        if event.button != self.button:
            return

        trans = self.axes.transData.inverted()
        x1, y1 = trans.transform_point((self.event_press.x, self.event_press.y))
        x2, y2 = trans.transform_point((self.event_release.x, self.event_release.y))

        # Order points
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Bound in axes limits
        lx1, lx2 = self.axes.get_xlim()
        ly1, ly2 = self.axes.get_ylim()
        x1 = max(lx1, min(lx2, x1))
        x2 = max(lx1, min(lx2, x2))
        y1 = max(ly1, min(ly2, y1))
        y2 = max(ly1, min(ly2, y2))

        self.extent = x1, x2, y1, y2

        if self.callback is not None:
            self.callback(self.event_press, self.event_release)
        self.hide()

    def activate(self, axes: Axes, callback: Callable = None) -> None:
        self.axes = axes
        self.callback = callback
        self.cids = [
            self.parent().mpl_connect("button_press_event", self._press),
            self.parent().mpl_connect("motion_notify_event", self._move),
            self.parent().mpl_connect("button_release_event", self._release),
        ]

    def deactivate(self) -> None:
        for cid in self.cids:
            self.parent().mpl_disconnect(cid)

    def close(self) -> None:
        self.deactivate()
        super().close()


