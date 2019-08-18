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
        vmin: float = None,
        vmax: float = None,
    ):
        self.cmap = cmap

        self._range = (vmin, vmax)
        self._ranges: Dict[str, Tuple[float, float]] = {}

    def set_cmap(self, name: str) -> None:
        self.cmap = self.COLORMAPS[name]

    def get_range(self, isotope: str = None) -> Tuple[float, float]:
        if isotope is None:
            return self._range
        return self._ranges.setdefault(isotope, self._range)

    def set_range(self, range: Tuple[float, float], isotope: str = None) -> None:
        if isotope is None:
            self._range = range
        self._ranges[isotope] = range


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
