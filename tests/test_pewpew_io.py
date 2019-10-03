import os.path

from pew.config import Config

from pewpew.lib.io import import_any


def test_import_any():
    data_path = os.path.join(os.path.dirname(__file__), "data", "io")

    paths = [
        os.path.join(data_path, f)
        for f in ["agilent.b", "csv.csv", "npz.npz", "thermo.csv"]
    ]

    lasers = import_any(paths, Config())
    assert len(lasers) == 4
