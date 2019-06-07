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
        bbox = (
            self.figure.get_tightbbox(
                self.get_renderer(),
                bbox_extra_artists=self.get_default_bbox_extra_artists(),
            )
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
