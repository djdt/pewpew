import numpy as np

import pytest

from pewpew.lib import calc


def test_otsu():
    np.random.seed(48150942)
    x = np.random.normal(scale=0.2, size=(100, 100))
    x[:, :50] += np.random.normal(loc=3.0, size=(100, 50))

    assert pytest.approx(calc.otsu(x), 1.5928213)
