import numpy as np

from typing import Tuple


def otsu(x: np.ndarray):
    # from scikit-learn
    hist, bin_edges = np.histogram(x, range=(x.min(), np.nanpercentile(x, 90)), bins=64)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]

    u1 = np.cumsum(hist * bin_centers) / w1
    u2 = (np.cumsum((hist * bin_centers)[::-1]) / w2[::-1])[::-1]

    i = np.argmax(w1[:-1] * w2[1:] * (u1[:-1] - u2[1:]) ** 2)
    return bin_centers[i]


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
