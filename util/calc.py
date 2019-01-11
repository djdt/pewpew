import numpy as np

from typing import Tuple


def rolling_window(x: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    x = np.ascontiguousarray(x)
    shape = tuple(np.array(x.shape) // window) + window
    strides = tuple(np.array(x.strides) * window) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def rolling_window2(x: np.ndarray, window: Tuple[int, int], step: int) -> np.ndarray:
    x = np.ascontiguousarray(x)
    slices = tuple(slice(None, None, st) for st in (step,) * x.ndim)
    index_strides = x[slices].strides
    win_index_shape = (((np.array(x.shape) - np.array(window)) // step) + 1)

    shape = tuple(list(win_index_shape) + list(window))
    strides = tuple(list(index_strides) + list(x.strides))
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def despike(x: np.ndarray, n: int = 3) -> np.ndarray:
    mean = np.mean(x)
    std = np.std(x)
    return x[np.abs(mean - x) < n * std]


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
    a = np.arange(110).reshape((11, 10))
    print(a)
    print(rolling_window2(a, (5, 5), 5))
