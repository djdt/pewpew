from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap

ppSpectral = LinearSegmentedColormap.from_list(
    "ppSpectral",
    [
        (0.10196078, 0.086275, 0.133333),
        (0.368627, 0.309804, 0.635294),
        (0.196078, 0.533333, 0.741176),
        (0.400000, 0.760784, 0.647059),
        (0.670588, 0.866667, 0.643137),
        (0.901961, 0.960784, 0.596078),
        (1.000000, 1.000000, 0.749020),
        (0.996078, 0.878431, 0.545098),
        (0.992157, 0.682353, 0.380392),
        (0.956863, 0.427451, 0.262745),
        (0.835294, 0.243137, 0.309804),
        (0.619608, 0.003922, 0.258824),
    ],
)

register_cmap("ppSpectral", cmap=ppSpectral)

COLORMAPS = [
    ("Magma", "magma", True, True, "Perceptually uniform colormap from R."),
    ("Viridis", "viridis", True, True, "Perceptually uniform colormap from R."),
    ("Cividis", "cividis", True, True, "Perceptually uniform colormap from R."),
    ("Blue Red", "RdBu_r", False, True, "Diverging colormap from colorbrewer."),
    (
        "Blue Yellow Red",
        "RdYlBu_r",
        False,
        True,
        "Diverging colormap from colorbrewer.",
    ),
    (
        "PP Spectral",
        "ppSpectral",
        False,
        False,
        "Custom rainbow colormap based on colorbrewer Spectral.",
    ),
]
