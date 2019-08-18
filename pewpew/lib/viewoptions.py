import numpy as np
from matplotlib.colors import Colormap

from pewpew.lib.mplcolors import ppSpectral

from typing import Dict, Tuple, Union


class ViewOptions(object):
    def __init__(self, *args, **kwargs) -> None:
        self.colors = ColorOptions()
        self.image = ImageOptions()
        self.font = FontOptions()

        self.calibrate = True
        self.units = "μm"


class ColorOptions(object):
    COLORMAPS = {
        "Magma": "magma",
        "Viridis": "viridis",
        "PewPew": ppSpectral,
        "Cividis": "cividis",
        "Blue Red": "RdBu_r",
        "Blue Yellow Red": "RdYlBu_r",
        "Grey": "grey",
    }
    COLORMAP_DESCRIPTIONS = {
        "Magma": "Perceptually uniform colormap from R.",
        "Viridis": "Perceptually uniform colormap from R.",
        "PewPew": "Custom colormap based on colorbrewers Spectral.",
        "Cividis": "Perceptually uniform colormap from R.",
        "Blue Red": "Diverging colormap from colorbrewer.",
        "Blue Yellow Red": "Diverging colormap from colorbrewer.",
        "Grey": "Simple black to white gradient.",
    }

    def __init__(
        self,
        cmap: Union[str, Colormap] = ppSpectral,
        default_range: Tuple[Union[float, str], Union[float, str]] = (0.0, "99%"),
    ):
        self.cmap = cmap

        self.default_range = default_range
        self._ranges: Dict[str, Tuple[Union[float, str], Union[float, str]]] = {}

    def set_cmap(self, name: str) -> None:
        self.cmap = self.COLORMAPS[name]

    def get_range(self, isotope: str) -> Tuple[Union[float, str], Union[float, str]]:
        return self._ranges.get(isotope, self.default_range)

    def set_range(
        self, range: Tuple[Union[float, str], Union[float, str]], isotope: str
    ) -> None:
        self._ranges[isotope] = range

    def get_range_as_float(self, isotope: str, data: np.ndarray) -> Tuple[float, float]:
        vmin, vmax = self.get_range(isotope)
        if isinstance(vmin, str):
            vmin = np.percentile(data, float(vmin.rstrip("%")))
        if isinstance(vmax, str):
            vmax = np.percentile(data, float(vmax.rstrip("%")))
        return vmin, vmax


class ImageOptions(object):
    INTERPOLATIONS = {"None": "none", "Bilinear": "bilinear", "Bicubic": "bicubic"}

    def __init__(self, interpolation: str = "none", alpha: float = 1.0):
        self.interpolation = interpolation
        self.alpha = alpha


class FontOptions(object):
    def __init__(self, size: int = 12, color: str = "white"):
        self.size = size
        self.color = color

    def props(self) -> dict:
        return {"size": self.size, "color": self.color}
