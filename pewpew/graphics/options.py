from PySide2 import QtCore, QtGui
import numpy as np

from typing import Dict, Tuple, Union


class GraphicsOptions(object):
    colortables = {
        "cividis": "Perceptually uniform colormap.",
        "grey": "Simple black to white gradient.",
        "magma": "Perceptually uniform colormap.",
        "inferno": "Perceptually uniform colormap.",
        # "PewPew": "Custom colormap based on colorbrewers Spectral.",
        "turbo": "Google's improved version of rainbow colormap jet.",
        "viridis": "Perceptually uniform colormap.",
    }

    def __init__(self, *args, **kwargs) -> None:
        # Todo: maybe alignments here?
        self.items = {
            "label": True,
            "scalebar": True,
            "colorbar": True,
        }

        self.colortable = "viridis"
        self._colorranges: Dict[str, Tuple[Union[float, str], Union[float, str]]] = {}
        self.colorrange_default = (0.0, "99%")

        self.smoothing = False

        self.font = QtGui.QFont()
        self.font.setPointSize(16)
        self.font_color = QtCore.Qt.white

        self.calibrate = True
        self.units = "μm"

    def get_colorrange(self, name: str) -> Tuple[Union[float, str], Union[float, str]]:
        return self._colorranges.get(name, self.colorrange_default)

    def get_colorrange_as_float(
        self, name: str, data: np.ndarray
    ) -> Tuple[float, float]:
        vmin, vmax = self.get_colorrange(name)
        if isinstance(vmin, str):
            vmin = np.nanpercentile(data, float(vmin.rstrip("%")))
        if isinstance(vmax, str):
            vmax = np.nanpercentile(data, float(vmax.rstrip("%")))
        return vmin, vmax  # type: ignore

    def get_colorrange_as_percentile(
        self, name: str, data: np.ndarray
    ) -> Tuple[float, float]:
        vmin, vmax = self.get_colorrange(name)
        if isinstance(vmin, str):
            vmin = float(vmin.rstrip("%"))
        else:
            vmin = np.count_nonzero(data < vmin) / data.size * 100
        if isinstance(vmax, str):
            vmax = float(vmax.rstrip("%"))
        else:
            vmin = np.count_nonzero(data < vmax) / data.size * 100
        return vmin, vmax  # type: ignore

    def set_colorrange(
        self, name: str, colorrange: Tuple[Union[float, str], Union[float, str]]
    ) -> None:
        self._colorranges[name] = colorrange
