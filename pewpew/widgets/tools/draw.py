import numpy as np
from PySide2 import QtCore, QtWidgets
from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent, PickEvent

from pewpew.lib.viewoptions import ViewOptions

from pewpew.widgets.canvases import InteractiveCanvas, LaserCanvas


class DrawCanvas(LaserCanvas, InteractiveCanvas):
    def __init__(
        self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None
    ) -> None:
        super().__init__(viewoptions=viewoptions, parent=parent)

        self.brush = {"shape": None, "size": 1, "value": np.nan}
        self.brush_button = 1
        self.move_button = 2

    def redrawFigure(self) -> None:
        super().redrawFigure()

    def ignore_event(self, event: LocationEvent) -> bool:
        if event.name in ["key_press_event"]:
            return True
        elif (
            event.name in ["button_press_event", "button_release_event"]
            and event.button != self.button
        ):
            return True

        return super().ignore_event(event)

    def press(self, event: MouseEvent) -> None:
        pass

    def release(self, event: MouseEvent) -> None:
        pass

    def move(self, event: MouseEvent) -> None:
        if (
            all(state in self.state for state in ["move", "zoom"])
            # and "selection" not in self.state
            and event.button == self.move_button
        ):
            x1, x2, y1, y2 = self.view_limits
            xmin, xmax, ymin, ymax = self.extent
            dx = self.eventpress.xdata - event.xdata
            dy = self.eventpress.ydata - event.ydata

            # Move in opposite direction to drag
            if x1 + dx > xmin and x2 + dx < xmax:
                x1 += dx
                x2 += dx
            if y1 + dy > ymin and y2 + dy < ymax:
                y1 += dy
                y2 += dy
            self.view_limits = x1, x2, y1, y2

        # Update the status bar
        x, y = event.xdata, event.ydata
        v = self.image.get_cursor_data(event)
        self.cursorMoved.emit(x, y, v)

    def scroll(self, event: MouseEvent) -> None:
        zoom_factor = 0.1 * event.step

        x1, x2, y1, y2 = self.view_limits

        x1 = x1 + (event.xdata - x1) * zoom_factor
        x2 = x2 - (x2 - event.xdata) * zoom_factor
        y1 = y1 + (event.ydata - y1) * zoom_factor
        y2 = y2 - (y2 - event.ydata) * zoom_factor

        if x1 > x2 or y1 > y2:
            return

        xmin, xmax, ymin, ymax = self.extent

        # If (un)zoom overlaps an edge attempt to shift it
        if x1 < xmin:
            x1, x2 = xmin, min(xmax, x2 + (xmin - x1))
        if x2 > xmax:
            x1, x2 = max(xmin, x1 - (x2 - xmax)), xmax

        if y1 < ymin:
            y1, y2 = ymin, min(ymax, y2 + (ymin - y1))
        if y2 > ymax:
            y1, y2 = max(ymin, y1 - (y2 - ymax)), ymax

        if (x1, x2, y1, y2) != self.extent:
            self.state.add("zoom")
        else:
            self.state.discard("zoom")
        self.view_limits = x1, x2, y1, y2

    def axis_enter(self, event: LocationEvent) -> None:
        pass

    def axis_leave(self, event: LocationEvent) -> None:
        try:
            status_bar = self.window().statusBar()
            status_bar.clearMessage()
        except AttributeError:
            pass
