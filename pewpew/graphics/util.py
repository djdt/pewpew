import numpy as np


def array_as_indexed8(
    array: np.ndarray, vmin: float = None, vmax: float = None
) -> np.ndarray:
    if vmin is None:
        vmin = np.amin(array)
    if vmax is None:
        vmax = np.amax(array)
    if vmin > vmax:
        vmin, vmax = vmax, vmin

    array = np.atleast_2d(array)
    data = np.clip(array, vmin, vmax)
    data = (data - vmin) / (vmax - vmin)
    return (data * 255.0).astype(np.uint8)
