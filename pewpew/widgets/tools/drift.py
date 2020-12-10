import copy
import numpy as np

from PySide2 import QtCore, QtWidgets

from matplotlib.artist import Artist
from matplotlib.backend_bases import PickEvent, MouseEvent
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.transforms import blended_transform_factory

from pewlib import Calibration

from pewpew.lib.numpyqt import NumpyArrayTableModel
from pewpew.lib.viewoptions import ViewOptions
from pewpew.lib.mpltools import LabeledLine2D
from pewpew.validators import DoubleSignificantFiguresDelegate
from pewpew.widgets.canvases import InteractiveImageCanvas
from pewpew.widgets.dialogs import CalibrationCurveDialog
from pewpew.widgets.modelviews import BasicTable, BasicTableView
from pewpew.widgets.laser import LaserWidget

from .tool import ToolWidget

from typing import Any, Dict, List, Tuple


class DriftCanvas(InteractiveImageCanvas):
    guidesChanged = QtCore.Signal()

    def __init__(self, viewoptions: ViewOptions, parent: QtWidgets.QWidget = None):
        super().__init__(widget_button=1, state=(), parent=parent)
        self.viewoptions = viewoptions

        self.connect_event("draw_event", self.update_background)
        self.connect_event("resize_event", self._resize)
        self.connect_event("pick_event", self._pick)

        self.guides_need_draw = True

        self.background = None
        self.edge_guides: List[Line2D] = []

        self.picked_artist: Artist = None

        self.drawFigure()

    def _resize(self, event) -> None:
        self.guides_need_draw = True

    def _pick(self, event: PickEvent) -> None:
        if self.ignore_event(event.mouseevent):
            return
        if event.mouseevent.button == self.widget_button:
            self.picked_artist = event.artist
        else:
            self.picked_artist = None

    def update_background(self, event) -> None:
        self.background = self.copy_from_bbox(self.ax.bbox)
        if self.guides_need_draw:
            self.blitGuides()

    def move(self, event: MouseEvent) -> None:
        if self.picked_artist is not None:
            self.picked_artist.set_xdata([event.xdata, event.xdata])
            self.blitGuides()

    def release(self, event: MouseEvent) -> None:
        if self.picked_artist is not None:
            x = self.picked_artist.get_xdata()[0]
            y = self.picked_artist.get_ydata()[0]
            shape = self.image.get_array().shape
            x0, x1, y0, y1 = self.extent
            # Snap to pixels
            pxy = (x1 - x0, y1 - y0) / np.array((shape[1], shape[0]))
            x, y = pxy * np.round((x, y) / pxy)

            self.picked_artist.set_xdata([x, x])

            self.blitGuides()
            self.guidesChanged.emit()
            self.picked_artist = None

    def drawData(
        self, data: np.ndarray, extent: Tuple[float, float, float, float]
    ) -> None:
        if self.image is not None:
            self.image.remove()
        self.image = self.ax.imshow(
            data,
            extent=extent,
            cmap=self.viewoptions.image.cmap,
            interpolation=self.viewoptions.image.interpolation,
            alpha=self.viewoptions.image.alpha,
            aspect="equal",
            origin="lower",
        )

    def drawEdgeGuides(self, xpos: Tuple[float, float]) -> None:
        for line in self.edge_guides:
            line.remove()
        self.edge_guides = []

        for x in xpos:
            line = Line2D(
                (x, x),
                (0.0, 1.0),
                transform=blended_transform_factory(
                    self.ax.transData, self.ax.transAxes
                ),
                color="white",
                linestyle="-",
                path_effects=[withStroke(linewidth=2.0, foreground="black")],
                linewidth=1.0,
                picker=True,
                pickradius=5,
                animated=True,
            )
            self.edge_guides.append(line)
            self.ax.add_artist(line)

    def blitGuides(self) -> None:
        if self.background is not None:
            self.restore_region(self.background)

        for a in self.edge_guides:
            self.ax.draw_artist(a)

        self.blit()
        self.guides_need_draw = False

    def getCurrentTrim(self) -> Tuple[int, int]:
        shape = self.image.get_array().shape
        x0, x1, y0, y1 = self.extent
        px = (x1 - x0) / shape[1]  # Data coords
        trim = np.array(
            [guide.get_xdata()[0] / px for guide in self.edge_guides], dtype=int
        )

        return np.min(trim), np.max(trim)

class DriftTool(ToolWidget):
    def __init__(self, widget: LaserWidget):
        super().__init__(widget, apply_all=True)
