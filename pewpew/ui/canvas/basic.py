from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from typing import List, Tuple


class BasicCanvas(FigureCanvasQTAgg):
    def __init__(
        self,
        figsize: Tuple[float, float] = (5.0, 5.0),
        parent: QtWidgets.QWidget = None,
    ):
        fig = Figure(frameon=False, tight_layout=True, figsize=figsize, dpi=100)
        super().__init__(fig)

        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

    def get_default_bbox_extra_artists(self) -> List:
        # Temporary until mpl fixes this
        bbox_artists = [
            artist
            for artist in self.figure.get_children()
            if (artist.get_visible() and artist.get_in_layout())
        ]
        for ax in self.figure.axes:
            if ax.get_visible():
                bbox_artists.extend(ax.get_default_bbox_extra_artists())
        # we don't want the figure's patch to influence the bbox calculation
        if self.figure.patch in bbox_artists:
            bbox_artists.remove(self.figure.patch)
        return bbox_artists

    def copyToClipboard(self) -> None:
        # The y axis is inverted so we must invert the bounds
        y = self.size().height()
        bbox = self.figure.get_tightbbox(
            self.renderer, bbox_extra_artists=self.get_default_bbox_extra_artists()
        ).transformed(self.figure.dpi_scale_trans)
        x0, y0, w, h = bbox.bounds
        # Qt and mpl coords differ
        y0 = y - y0 - h
        QtWidgets.QApplication.clipboard().setPixmap(
            self.grab(QtCore.QRect(int(x0), int(y0), int(w), int(h)))
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(250, 250)
