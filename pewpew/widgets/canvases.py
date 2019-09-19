import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets

from matplotlib.backend_bases import KeyEvent, MouseEvent, LocationEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patheffects import Normal, SimpleLineShadow
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import AxesWidget, RectangleSelector

from mpl_toolkits.axes_grid1 import make_axes_locatable

from laserlib.laser import Laser
from laserlib.krisskross import KrissKrossConfig

from pewpew.lib.mpltools import MetricSizeBar, image_extent_to_data
from pewpew.lib.mplwidgets import (
    _ImageSelectionWidget,
    RectangleImageSelectionWidget,
    LassoImageSelectionWidget,
)
from pewpew.lib.viewoptions import ViewOptions

from typing import Callable, List, Tuple


class BasicCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        figsize: Tuple[float, float] = (5.0, 5.0),
        parent: QtWidgets.QWidget = None,
    ):
        fig = Figure(frameon=False, tight_layout=True, figsize=figsize)
        super().__init__(fig)

        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

    def copyToClipboard(self) -> None:
        bbox = (
            self.figure.get_tightbbox(self.get_renderer())
            .transformed(self.figure.dpi_scale_trans)
            .padded(5)  # Pad to look nicer
        )
        (x0, y0), (x1, y1) = bbox.get_points().astype(int)
        ymax = self.size().height()  # We need to invert for mpl to Qt
        QtWidgets.QApplication.clipboard().setPixmap(
            self.grab(QtCore.QRect(x0, ymax - y1, x1 - x0, y1 - y0))
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 250)


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


class LaserCanvas(BasicCanvas):
    def __init__(
        self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None
    ) -> None:
        super().__init__(parent=parent)
        self.viewoptions = viewoptions

        self.redrawFigure()
        self.image: AxesImage = None

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        if self.image is None:
            return (0, 0, 0, 0)
        return self.image.get_extent()

    @property
    def view_limits(self) -> Tuple[float, float, float, float]:
        x0, x1, = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        return x0, x1, y0, y1

    @view_limits.setter
    def view_limits(self, limits: Tuple[float, float, float, float]) -> None:
        x0, x1, y0, y1 = limits
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.draw_idle()

    def redrawFigure(self) -> None:
        # Restore view limits
        self.figure.clear()
        self.ax = self.figure.subplots()
        self.ax.set_facecolor("black")
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.autoscale(False)
        if self.viewoptions.canvas.colorbar:
            div = make_axes_locatable(self.ax)
            self.cax = div.append_axes(
                self.viewoptions.canvas.colorbarpos, size=0.1, pad=0.05
            )

    def drawColorbar(self, label: str) -> None:
        self.cax.clear()
        if self.viewoptions.canvas.colorbarpos in ["right", "left"]:
            orientation = "vertical"
        else:
            orientation = "horizontal"
        self.figure.colorbar(
            self.image,
            label=label,
            ax=self.ax,
            cax=self.cax,
            orientation=orientation,
            ticks=MaxNLocator(nbins=6),
        )
        self.cax.tick_params(labelsize=self.viewoptions.font.size)

    def drawData(
        self,
        data: np.ndarray,
        extent: Tuple[float, float, float, float],
        isotope: str = None,
    ) -> None:
        self.ax.clear()

        # Calculate the range
        vmin, vmax = self.viewoptions.colors.get_range_as_float(isotope, data)
        # Ensure than vmin is actually lower than vmax
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        # Cmap bad to red
        # Plot the image
        self.image = self.ax.imshow(
            data,
            cmap=self.viewoptions.image.cmap,
            interpolation=self.viewoptions.image.interpolation,
            alpha=self.viewoptions.image.alpha,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            origin="upper",
        )

    def drawLaser(self, laser: Laser, name: str, layer: int = None) -> None:
        # Get the trimmed and calibrated data
        kwargs = {"calibrate": self.viewoptions.calibrate, "layer": layer, "flat": True}

        data = laser.get(name, **kwargs)
        unit = (
            str(laser.data[name].calibration.unit) if self.viewoptions.calibrate else ""
        )

        # Get extent
        extent = (
            laser.config.data_extent(data, layer)
            if isinstance(laser.config, KrissKrossConfig)
            else laser.config.data_extent(data)
        )

        # Only change the view if new or the laser extent has changed (i.e. conf edit)
        if self.extent != extent:
            self.view_limits = extent

        # If data is empty create a dummy data
        if data is None or data.size == 0:
            data = np.array([[0]], dtype=np.float64)

        # Restor any view limit
        view_limits = self.view_limits
        self.drawData(data, extent, name)
        self.view_limits = view_limits

        if self.viewoptions.canvas.colorbar:
            self.drawColorbar(unit)

        if self.viewoptions.canvas.label:
            text = AnchoredText(
                name,
                "upper left",
                pad=0.2,
                borderpad=0.1,
                frameon=False,
                prop=self.viewoptions.font.props(),
            )
            self.ax.add_artist(text)

        if self.viewoptions.canvas.scalebar:
            scalebar = MetricSizeBar(
                self.ax,
                loc="upper right",
                color=self.viewoptions.font.color,
                font_properties=self.viewoptions.font.mpl_props(),
            )
            self.ax.add_artist(scalebar)


class InteractiveLaserCanvas(LaserCanvas, InteractiveCanvas):
    def __init__(
        self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None
    ) -> None:
        super().__init__(viewoptions=viewoptions, parent=parent)

        self.state = set(["move"])
        self.button = 1

        shadow = self.palette().color(QtGui.QPalette.Shadow)
        highlight = self.palette().color(QtGui.QPalette.Highlight)
        lineshadow = SimpleLineShadow(
            offset=(0.5, -0.5), alpha=0.66, shadow_color=shadow.name()
        )
        self.rectprops = {
            "edgecolor": highlight.name(),
            "facecolor": "none",
            "linestyle": "-",
            "linewidth": 1.1,
            "path_effects": [lineshadow, Normal()],
        }
        self.lineprops = {
            "color": highlight.name(),
            "linestyle": "--",
            "linewidth": 1.1,
            "path_effects": [lineshadow, Normal()],
        }
        self.mask_rgba = np.array(
            [highlight.red(), highlight.green(), highlight.blue(), 255 * 0.5],
            dtype=np.uint8,
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.clearSelection()
        super().keyPressEvent(event)

    def redrawFigure(self) -> None:
        super().redrawFigure()
        if self.widget is not None:
            self.widget.ax = self.ax

    def drawLaser(self, laser: Laser, name: str, layer: int = None) -> None:
        super().drawLaser(laser, name, layer)
        # Save some variables for the status bar
        self.px, self.py = (
            (laser.config.get_pixel_width(layer), laser.config.get_pixel_height(layer))
            if isinstance(laser.config, KrissKrossConfig)
            else (laser.config.get_pixel_width(), laser.config.get_pixel_height())
        )
        self.ps = laser.config.speed

    def endSelection(self) -> None:
        if self.widget is not None:
            self.widget.set_active(False)

    def startLassoSelection(self) -> None:
        self.clearSelection()
        self.state.add("selection")
        self.widget = LassoImageSelectionWidget(
            self.image,
            self.mask_rgba,
            useblit=True,
            button=self.button,
            lineprops=self.lineprops,
        )
        self.widget.set_active(True)
        self.setFocus(QtCore.Qt.NoFocusReason)

    def startRectangleSelection(self) -> None:
        self.clearSelection()
        self.state.add("selection")
        self.widget = RectangleImageSelectionWidget(
            self.image,
            self.mask_rgba,
            useblit=True,
            button=self.button,
            lineprops=self.lineprops,
        )
        self.widget.set_active(True)
        self.setFocus(QtCore.Qt.NoFocusReason)

    def clearSelection(self) -> None:
        if self.widget is not None:
            self.widget.set_active(False)
            self.widget.set_visible(False)
            self.widget.update()
            self.draw_idle()
        self.state.discard("selection")
        self.widget = None

    def getSelection(self) -> np.ndarray:
        if self.widget is not None and isinstance(self.widget, _ImageSelectionWidget):
            return self.widget.mask
        else:
            return None

    def getMaskedData(self) -> np.ndarray:
        x0, x1, y0, y1 = self.view_limits
        (x0, y0), (x1, y1) = (
            image_extent_to_data(self.image).transform(((x0, y1), (x1, y0))).astype(int)
        )
        data = self.image.get_array()[y0:y1, x0:x1]  # type: ignore
        mask = self.getSelection()
        if mask is not None and not np.all(mask == 0):
            mask = mask[y0:y1, x0:x1]
            # Trim out nan rows and columns
            data = np.where(mask, data, np.nan)
            data = data[:, ~np.isnan(data).all(axis=0)]
            data = data[~np.isnan(data).all(axis=1)]
        return data

    def ignore_event(self, event: LocationEvent) -> bool:
        if event.name in ["scroll_event", "key_press_event"]:
            return True
        elif (
            event.name in ["button_press_event", "button_release_event"]
            and event.button != self.button
        ):
            return True

        if event.inaxes != self.ax:
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
            and event.button == self.button
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
        try:
            status_bar = self.window().statusBar()
            x, y = event.xdata, event.ydata
            v = self.image.get_cursor_data(event)
            unit = self.viewoptions.units
            if unit == "row":
                x, y = int(x / self.px), int(y / self.py)
            elif unit == "second":
                x = event.xdata / self.ps
                y = 0
            if np.isfinite(v):
                status_bar.showMessage(f"{x:.4g},{y:.4g} [{v:.4g}]")
            else:
                status_bar.showMessage(f"{x:.4g},{y:.4g} [nan]")
        except AttributeError:
            pass

    def axis_enter(self, event: LocationEvent) -> None:
        pass

    def axis_leave(self, event: LocationEvent) -> None:
        try:
            status_bar = self.window().statusBar()
            status_bar.clearMessage()
        except AttributeError:
            pass

    def startZoom(self) -> None:
        self.clearSelection()
        self.widget = RectangleSelector(
            self.ax,
            self.zoom,
            useblit=True,
            drawtype="box",
            button=self.button,
            rectprops=self.rectprops,
        )
        self.widget.set_active(True)

    def zoom(self, press: MouseEvent, release: MouseEvent) -> None:
        self.clearSelection()
        self.view_limits = (press.xdata, release.xdata, press.ydata, release.ydata)
        self.state.add("zoom")

    def unzoom(self) -> None:
        self.state.discard("zoom")
        self.view_limits = self.extent
