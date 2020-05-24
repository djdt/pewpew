import os.path
import warnings

from pew.config import Config

from pew.io.error import PewWarning

from pewpew.lib.io import import_any


def test_import_any():
    data_path = os.path.join(os.path.dirname(__file__), "data", "io")

    paths = [
        os.path.join(data_path, f)
        for f in ["agilent.b", "csv.csv", "npz.npz", "thermo.csv"]
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PewWarning)
        lasers = import_any(paths, Config())
        assert len(lasers) == 4
