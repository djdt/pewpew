import numpy as np

# from pewpew.lib import convolve
from pewpew.lib.calc import view_as_blocks

from typing import Tuple


# def low_pass_filter(
#     x: np.ndarray, block: Tuple[int, int], threshold: float = 0.5
# ) -> np.ndarray:
#     kernel = convolve.normal(block[0], block[0] / 2.0, 0.0)[:, 1]
#     kernel /= kernel.sum()
#     y = np.apply_along_axis(np.convolve, 0, x, kernel, mode="same")
#     y = np.apply_along_axis(np.convolve, 1, y, kernel, mode="same")

#  # NO NO NO NO
#     diff = np.divide(np.abs(x - y), np.abs(x), where=x != 0)
#     diff[x == 0] = 0.0
#     diff /= diff.max()

#     return np.where(diff > threshold, y, x)


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

    # As the mean is sensitive to outliers reclaculate it
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

    # Outliers are n medians from data
    outliers = np.abs(x - medians) > threshold * median_medians

    return np.where(outliers, medians, x)
