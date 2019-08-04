from PySide2 import QtWidgets

from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent
from matplotlib.widgets import AxesWidget

from pewpew.ui.canvas.basic import BasicCanvas

from typing import Callable, List, Tuple


class InteractiveCanvas(BasicCanvas):
    def __init__(
        self,
        figsize: Tuple[float, float] = (5.0, 5.0),
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(figsize, parent)

        self.cids: List[int] = []
        self.default_events = {
            "axis_enter_event": self._axis_enter,
            "axis_leave_event": self._axis_leave,
            "key_press_event": self._keypress,
            "button_press_event": self._press,
            "button_release_event": self._release,
            "motion_notify_event": self._move,
            "scroll_event": self._scroll,
        }

        for event, callback in self.default_events.items():
            self.connect_event(event, callback)

        self.widget: AxesWidget = None

    def close(self) -> None:
        self.disconnect_events()
        super().close()

    def connect_event(self, event: str, callback: Callable) -> None:
        self.cids.append(self.mpl_connect(event, callback))

    def disconnect_events(self) -> None:
        for cid in self.cids:
            self.mpl_disconnect(cid)
        self.cids.clear()

    def ignore_event(self, event: MouseEvent) -> bool:
        if self.widget is not None and self.widget.get_active():
            return True
        return False

    def _axis_enter(self, event: LocationEvent) -> None:
        if self.ignore_event(event):
            return
        self.axis_enter(event)

    def _axis_leave(self, event: LocationEvent) -> None:
        if self.ignore_event(event):
            return
        self.axis_leave(event)

    def _keypress(self, event: KeyEvent) -> None:
        if self.ignore_event(event):
            return
        self.keypress(event)

    def _press(self, event: MouseEvent) -> None:
        if self.ignore_event(event):
            return
        self.eventpress = event
        self.press(event)

    def _release(self, event: MouseEvent) -> None:
        if self.ignore_event(event):
            return
        self.eventrelease = event
        self.release(event)

    def _move(self, event: MouseEvent) -> None:
        if self.ignore_event(event):
            return
        self.move(event)

    def _scroll(self, event: MouseEvent) -> None:
        if self.ignore_event(event):
            return
        self.scroll(event)
