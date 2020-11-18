import warnings
from pathlib import Path

import pytest

from pew.config import Config

from pew.io.error import PewWarning, PewException

from pewpew.lib.io import import_any


def test_import_any():
    path = Path(__file__).parent.joinpath("data", "io")

    paths = [path.joinpath(f) for f in ["test.b", "test.csv", "test.npz"]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PewWarning)
        lasers = import_any(paths, Config())
        assert len(lasers) == 3
