import numpy as np
from matplotlib.colors import Colormap
from matplotlib.patheffects import withStroke
from matplotlib.font_manager import FontProperties

from pewpew.lib.mplcolors import ppSpectral, googleTurbo

from typing import Dict, Tuple, Union


class ViewOptions(object):
    def __init__(self, *args, **kwargs) -> None:
        self.canvas = CanvasOptions()
        self.colors = ColorOptions()
        self.font = FontOptions()
        self.image = ImageOptions()

        self.calibrate = True
        self.units = "μm"


class CanvasOptions(object):
    def __init__(
        self, colorbar: bool = True, scalebar: bool = True, label: bool = True
    ):
        self.colorbar = colorbar
        self.colorbarpos = "bottom"
        self.scalebar = scalebar
        self.label = label


class ColorOptions(object):
    def __init__(
        self, default_range: Tuple[Union[float, str], Union[float, str]] = (0.0, "99%")
    ):
        self.default_range = default_range
        self._ranges: Dict[str, Tuple[Union[float, str], Union[float, str]]] = {}

    def get_range(self, isotope: str) -> Tuple[Union[float, str], Union[float, str]]:
        return self._ranges.get(isotope, self.default_range)

    def set_range(
        self, range: Tuple[Union[float, str], Union[float, str]], isotope: str
    ) -> None:
        self._ranges[isotope] = range

    def get_range_as_float(self, isotope: str, data: np.ndarray) -> Tuple[float, float]:
        vmin, vmax = self.get_range(isotope)
        if isinstance(vmin, str):
            vmin = np.nanpercentile(data, float(vmin.rstrip("%")))
        if isinstance(vmax, str):
            vmax = np.nanpercentile(data, float(vmax.rstrip("%")))
        return vmin, vmax  # type: ignore


class ImageOptions(object):
    COLORMAPS = {
        "Blue Red": "RdBu_r",
        "Blue Yellow Red": "RdYlBu_r",
        "Cividis": "cividis",
        "Grey": "gray",
        "Magma": "magma",
        "PewPew": ppSpectral,
        "Turbo": googleTurbo,
        "Viridis": "viridis",
    }
    COLORMAP_DESCRIPTIONS = {
        "Blue Red": "Diverging colormap from colorbrewer.",
        "Blue Yellow Red": "Diverging colormap from colorbrewer.",
        "Cividis": "Perceptually uniform colormap from R.",
        "Grey": "Simple black to white gradient.",
        "Magma": "Perceptually uniform colormap from R.",
        "PewPew": "Custom colormap based on colorbrewers Spectral.",
        "Turbo": "Google's improved version of rainbow colormap jet.",
        "Viridis": "Perceptually uniform colormap from R.",
    }
    INTERPOLATIONS = {
        "None": "none",
        "Bilinear": "bilinear",
        "Bicubic": "bicubic",
        "Gaussian": "gaussian",
    }

    def __init__(
        self,
        cmap: Union[str, Colormap] = ppSpectral,
        interpolation: str = "none",
        alpha: float = 1.0,
    ):
        self.cmap = cmap
        self.interpolation = interpolation
        self.alpha = alpha

    def set_cmap_name(self, name: str) -> None:
        self.cmap = self.COLORMAPS[name]

    def get_cmap_name(self) -> str:
        for k, v in self.COLORMAPS.items():
            if v == self.cmap:
                return k
        return ""

    def get_interpolation_name(self) -> str:
        for k, v in self.INTERPOLATIONS.items():
            if v == self.interpolation:
                return k
        return ""


class FontOptions(object):
    def __init__(self, size: int = 12, color: str = "white"):
        self.size = size
        self.color = color
        self.path_effects = [withStroke(linewidth=1.5, foreground="black")]

    def set_size(self, size: int) -> None:
        self.size = size

    def props(self) -> dict:
        return {
            "size": self.size,
            "color": self.color,
            "path_effects": self.path_effects,
            "font_properties": self.mpl_props()
        }

    def mpl_props(self) -> FontProperties:
        return FontProperties(size=self.size)
