import numpy as np


def weighted_rsq(x, y, w=None):
    c = np.cov(x, y, aweights=w)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c[0, 1]**2


def weighted_linreg(x, y, w=None):
    m, b = np.polyfit(x, y, 1, w=w)
    r2 = weighted_rsq(x, y, w)
    return m, b, r2


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4])
    b = np.array([10, 30, 30, 40])
    w = np.array([1, 0.1, 1, 1])

    print('good', weighted_linreg(a, a))
    print('bad', weighted_linreg(a, b))
    print('bad weighted', weighted_linreg(a, b, w))
