import os.path
import warnings

import pytest

from pew.config import Config

from pew.io.error import PewWarning, PewException

from pewpew.lib.io import import_any


def test_import_any():
    data_path = os.path.join(os.path.dirname(__file__), "data", "io")

    paths = [
        os.path.join(data_path, f)
        for f in ["agilent.b", "csv.csv", "npz.npz", "icap_rows.csv", "txt.txt"]
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PewWarning)
        lasers = import_any(paths, Config())
        assert len(lasers) == 5

    with pytest.raises(PewException):
        import_any(
            [os.path.join(os.path.dirname(__file__), "laser_canvas_raw.png")], Config()
        )
