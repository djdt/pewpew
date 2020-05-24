import numpy as np

from pewpew.lib import filters


def test_mean_filter():
    x = np.random.random((50, 50))
    x[10, 10] = 2.0
    x[10, 20] = -1.0

    # Nothing filtered when threshold is high
    f = filters.mean_filter(x, (5, 5), threshold=100.0)
    assert np.all(x == f)

    f = filters.mean_filter(x, (5, 5), threshold=3.0)
    assert np.allclose(f[10, 10], np.nanmean(np.where(x <= 1.0, x, np.nan)[8:13, 8:13]))
    assert np.allclose(f[10, 20], np.nanmean(np.where(x >= 0.0, x, np.nan)[8:13, 18:23]))


def test_median_filter():
    x = np.random.random((50, 50))
    x[10, 10] = 2.0
    x[10, 20] = -1.0

    # Nothing filtered when threshold is high
    f = filters.median_filter(x, (5, 5), threshold=100.0)
    assert np.all(x == f)

    f = filters.median_filter(x, (5, 5), threshold=3.0)
    assert np.allclose(f[10, 10], np.median(x[8:13, 8:13]))
    assert np.allclose(f[10, 20], np.median(x[8:13, 18:23]))
