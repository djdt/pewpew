from typing import List

import numpy as np
import pytest
from PySide6 import QtCore


@pytest.fixture(scope="session", autouse=True)
def clear_settings():
    QtCore.QSettings().clear()


def linear_data(names: str | list[str]) -> np.ndarray:
    names = names if isinstance(names, list) else [names]
    dtype = [(name, float) for name in names]
    data = np.empty((10, 10), dtype=dtype)
    for name in names:
        data[name] = np.repeat(
            np.arange(0, 10, dtype=float).reshape((10, 1)), 10, axis=1
        )
    return data


def rand_data(names: str | list[str]) -> np.ndarray:
    names = names if isinstance(names, list) else [names]
    dtype = [(name, float) for name in names]
    data = np.empty((10, 10), dtype=dtype)
    for name in names:
        data[name] = np.random.random((10, 10))
    return data
