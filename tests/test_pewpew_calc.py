import numpy as np

from pewpew.lib import calc


def test_greyscale_to_rgb():
    grey = calc.greyscale_to_rgb(np.linspace(0, 1, 5), [128, 256, 256])
    assert np.allclose(
        grey, [[0, 0, 0], [32, 64, 64], [64, 128, 128], [96, 192, 192], [128, 256, 256]]
    )


def test_kmeans():
    x = np.empty((100, 2), dtype=float)
    x[:50, 0] = np.random.normal(loc=-1.0, size=50)
    x[50:, 0] = np.random.normal(loc=1.0, size=50)
    x[:, 1] = np.random.normal(loc=3.0, size=100)

    x = np.random.random(100)
    y = np.zeros(100)
    y[50:] += 1.0
    y[80:] += 1.0

    idx = calc.kmeans(np.stack((x, y), axis=1), 3, init="kmeans++")

    _, counts = np.unique(idx, return_counts=True)
    assert np.allclose(np.sort(counts), [20, 30, 50], atol=2)


def test_kmeans_threshold():
    x = np.zeros(100)
    x[:25] += 1.0
    x[:50] += 1.0
    x[:75] += 1.0

    t = calc.kmeans_threshold(x, 4)

    assert np.allclose(t, [1.0, 2.0, 3.0])


def test_normalise():
    x = np.random.random(100)
    x = calc.normalise(x, -1.0, 2.33)
    assert x.min() == -1.0
    assert x.max() == 2.33


def test_otsu():
    x = np.hstack(
        (np.random.normal(1.0, 1.0, size=500), np.random.normal(4.0, 2.0, size=500))
    )

    assert np.allclose(calc.otsu(x), 3.0, atol=2e-1)


def test_shuffle_blocks():
    x = np.random.random((100, 100))
    m = np.zeros((100, 100))
    m[:52] = 1.0

    y = calc.shuffle_blocks(x, (5, 20), mask=m, mask_all=True)

    assert np.allclose(y[50:], x[50:])
    assert not np.allclose(y[:50], x[:50])
    assert np.allclose(y.sum(), x.sum())
