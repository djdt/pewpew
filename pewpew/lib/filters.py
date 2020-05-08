import numpy as np

from pewpew.lib.calc import view_as_blocks

from typing import Tuple


# def rolling_mean_filter(
#     x: np.ndarray, block: Tuple[int, int], threshold: float = 3.0
# ) -> np.ndarray:
#     blocks = view_as_blocks(x, block, (1, 1))
#     means = np.mean(blocks, axis=(2, 3))
#     stds = np.std(blocks, axis=(2, 3))
#     # Distance from mean (in stdevs)
#     diffs = np.divide(np.abs(blocks - means), stds, where=stds != 0.0)

#     # diffs = np.abs(blocks - means) / stds
#     # blocks[diffs > threshold] = np.nan
#     # Recalculate means without outliers
#     means = np.nanmean(blocks, axis=(2, 3))

#     return np.where(diffs > threshold, means, x)


def mean_filter(
    x: np.ndarray, block: Tuple[int, int], threshold: float = 3.0
) -> np.ndarray:
    """Rolling filter of size 'block'.
    If the value of x is 'threshold' stddevs from the local mean it is considered an outlier.
    Outliers are replaced with the local mean (excluding outliers).
    """
    # Prepare array by padding with nan
    px, py = block[0] // 2, block[1] // 2
    x_pad = np.pad(x, ((px, px), (py, py)), constant_values=np.nan)

    blocks = view_as_blocks(x_pad, block, (1, 1))
    # Calculate means and stds
    means = np.nanmean(blocks, axis=(2, 3))
    stds = np.nanstd(blocks, axis=(2, 3))
    # Check for outlying values and set as nan
    outliers = np.abs(x - means) > threshold * stds
    x_pad[px:-px, py:-py][outliers] = np.nan

    # Recalculate means without outliers
    means = np.nanmean(blocks, axis=(2, 3))

    return np.where(outliers, means, x)


def median_filter(
    x: np.ndarray, block: Tuple[int, int], threshold: float = 3.0
) -> np.ndarray:
    """Rolling filter of size 'block'.
    If the value of x is 'threshold' medians from the local median it is considered an outlier.
    Outliers are replaced with the local median.
    """
    # Prepare array by padding with nan
    px, py = block[0] // 2, block[1] // 2
    x_pad = np.pad(x, ((px, px), (py, py)), constant_values=np.nan)

    blocks = view_as_blocks(x_pad, block, (1, 1))
    # Calculate median and differences
    medians = np.nanmedian(blocks, axis=(2, 3))
    x_pad[px:-px, py:-py] = np.abs(x - medians)
    # Median of differences
    median_medians = np.nanmedian(blocks, axis=(2, 3))

    # Outliers are n medians from
    outliers = np.abs(x - medians) > threshold * median_medians

    return np.where(outliers, medians, x)


# def rolling_median_filter(
#     x: np.ndarray, window: Tuple[int, int], threshold: int = 3
# ) -> np.ndarray:
#     """Rolling median filter an array.

#     The window size should be an integer divisor of the array size.

#     Args:
#         window: Shape of the rolling window.
#         threshold: N-distance's from median to be considered outlier.

#     """
#     x = x.copy()
#     # Create view
#     roll = rolling_window(x, window)
#     # Distance from the median
#     medians = np.median(roll, axis=(2, 3), keepdims=True)
#     diffs = np.abs(roll - medians)
#     # Median difference
#     median_diffs = np.median(diffs, axis=(2, 3), keepdims=True)
#     # Normalise differences
#     diffs = np.divide(diffs, median_diffs, where=median_diffs != 0)
#     # Replace all over threshold and copy back into view
#     np.copyto(roll, medians, where=diffs > threshold)

#     return x
