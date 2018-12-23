import numpy as np

from typing import Tuple


def weighted_rsq(x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> float:
    c = np.cov(x, y, aweights=w)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return float(c[0, 1] ** 2)


def weighted_linreg(
    x: np.ndarray, y: np.ndarray, w: np.ndarray = None
) -> Tuple[float, float, float]:
    m, b = np.polyfit(x, y, 1, w=w)
    r2 = weighted_rsq(x, y, w)
    return m, b, r2


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5, 6, 7])
    b = np.array([5, 10, 14, 21, 28, 29, 34])
    w = np.array([1, 1, 1, 1, 0.1, 1, 1])

    print("non weighted", weighted_linreg(a, b))
    print("r2 pass", round(weighted_linreg(a, b)[2], 10) == 0.9817581301)
    print("weighted", weighted_linreg(a, b, w))
    print("r2 pass", round(weighted_linreg(a, b, w)[2], 10) == 0.9939799307)
