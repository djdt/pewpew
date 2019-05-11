import numpy as np

from typing import Tuple


def rolling_mean_filter(
    x: np.ndarray, window: Tuple[int, int], threshold: int = 3
) -> np.ndarray:
    """Rolling mean filter an array.

    The window size should be an integer divisor of the array size.

    Args:
        window: Shape of the rolling window.
        threshold: σ's value must be from mean to be an outlier.

    """
    x = x.copy()
    # Create view
    roll = rolling_window(x, window)
    # Distance from mean (in stdevs)
    means = np.mean(roll, axis=(2, 3), keepdims=True)
    stds = np.std(roll, axis=(2, 3), keepdims=True)
    diffs = np.abs(roll - means) / stds
    # Recalculate mean, without outliers
    roll[diffs > threshold] = np.nan
    means = np.nanmean(roll, axis=(2, 3), keepdims=True)
    # Replace all outliers and copy back into view
    np.copyto(roll, means, where=diffs > threshold)
    return x


def rolling_median_filter(
    x: np.ndarray, window: Tuple[int, int], threshold: int = 3
) -> np.ndarray:
    """Rolling median filter an array.

    The window size should be an integer divisor of the array size.

    Args:
        window: Shape of the rolling window.
        threshold: N-distance's from median to be considered outlier.

    """
    x = x.copy()
    # Create view
    roll = rolling_window(x, window)
    # Distance from the median
    medians = np.median(roll, axis=(2, 3), keepdims=True)
    diffs = np.abs(roll - medians)
    # Median difference
    median_diffs = np.median(diffs, axis=(2, 3), keepdims=True)
    # Normalise differences
    diffs = np.divide(diffs, median_diffs, where=median_diffs != 0)
    # Replace all over threshold and copy back into view
    np.copyto(roll, medians, where=diffs > threshold)

    return x


def rolling_window(x: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    """Create non-overlapping views into a array.

    Args:
        x: The array.
        window: The size of the view.

    Returns:
        An array of views.
    """
    x = np.ascontiguousarray(x)
    shape = tuple(np.array(x.shape) // window) + window
    strides = tuple(np.array(x.strides) * window) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def rolling_window_step(
    x: np.ndarray, window: Tuple[int, int], step: int
) -> np.ndarray:
    """Create overlapping views into a array.

    Args:
        x: The array.
        window: The size of the view.
        step: Offset of the next window.

    Returns:
        An array of views.
    """
    x = np.ascontiguousarray(x)
    slices = tuple(slice(None, None, st) for st in (step,) * x.ndim)
    shape = tuple(
        list(((np.array(x.shape) - np.array(window)) // np.array(step)) + 1)
        + list(window)
    )
    strides = tuple(list(x[slices].strides) + list(x.strides))
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


# def subpixel_offset(
#     x: np.ndarray, offsets: List[Tuple[int, int]], pixelsize: Tuple[int, int]
# ) -> np.ndarray:
#     """Takes a 3d array and stretches and offsets each layer.

#     Given an offset of (1,1) and pixelsize of (2,2) each layer will be streched by 2
#     and every even layer will be shifted by 1 pixel.

#     Args:
#         offsets: The pixel offsets in (x, y).
#         pixelsize: Final size to stretch to.

#     Returns:
#         The offset array.
#     """
#     # Offset for first layer must be zero
#     if offsets[0] != (0, 0):
#         offsets.insert(0, (0, 0))
#     overlap = np.max(offsets, axis=0)

#     if x.ndim != 3:
#         raise ValueError("Data must be three dimensional!")

#     # Calculate new shape
#     new_shape = np.array(x.shape[:2]) * pixelsize + overlap
#     # Create empty array to store data in
#     data = np.zeros((*new_shape, x.shape[2]), dtype=x.dtype)

#     for i in range(0, x.shape[2]):
#         # Cycle through offsets
#         start = offsets[i % len(offsets)]
#         end = -(overlap[0] - start[0]) or None, -(overlap[1] - start[1]) or None
#         # Stretch arrays as required
#         data[start[0] : end[0], start[1] : end[1], i] = np.repeat(
#             x[:, :, i], pixelsize[0], axis=0
#         ).repeat(pixelsize[1], axis=1)

#     return data


# def subpixel_offset_equal(
#     x: np.ndarray, offsets: List[int], pixelsize: int
# ) -> np.ndarray:
#     return subpixel_offset(x, [(o, o) for o in offsets], (pixelsize, pixelsize))


def weighting(x: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "x":
        return x
    if weighting == "1/x":
        return 1.0 / x
    elif weighting == "1/(x^2)":
        return 1.0 / (x ** 2.0)
    else:  # Default is no weighting
        return None


def weighted_rsq(x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> float:
    c = np.cov(x, y, aweights=w)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c[0, 1] ** 2.0


def weighted_linreg(
    x: np.ndarray, y: np.ndarray, w: np.ndarray = None
) -> Tuple[float, float, float]:
    m, b = np.polyfit(x, y, 1, w=w)
    r2 = weighted_rsq(x, y, w)
    return m, b, r2


if __name__ == "__main__":
    a = np.random.randint(0, 100, size=110).reshape(11, 10).astype(float)
    a = np.zeros((10, 10), dtype=float)

    rolling_median_filter(a, (5, 5))
