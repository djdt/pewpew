import numpy as np


def test_kmeans_threshold():
    x = np.zeros(100)
    x[:25] += 1.0
    x[:50] += 1.0
    x[:75] += 1.0

    t = kmeans_threshold(x, 4)

    assert np.allclose(t, [1.0, 2.0, 3.0])

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
    assert np.allclose(np.sort(counts), [20, 30, 50], atol=5)
    idx = calc.kmeans(np.stack((x, y), axis=1), 3, init="random")
    _, counts = np.unique(idx, return_counts=True)
    assert np.allclose(np.sort(counts), [20, 30, 50], atol=10)
