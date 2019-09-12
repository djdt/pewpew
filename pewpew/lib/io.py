import os.path

from laserlib import io
from laserlib import Laser, LaserConfig

from typing import List


def import_any(paths: List[str], config: LaserConfig) -> List[Laser]:
    lasers = []
    for path in paths:
        base, ext = os.path.splitext(path)
        name = os.path.basename(base)
        ext = ext.lower()
        if ext == ".npz":
            lasers.extend(io.npz.load(path))
        else:
            if ext == ".csv":
                try:
                    data = io.thermo.load(path)
                except io.error.LaserLibException:
                    data = io.csv.load(path)
            elif ext in [".txt", ".text"]:
                data = io.csv.load(path)
            elif ext == ".b":
                data = io.agilent.load(path)
            else:
                raise io.error.LaserLibException(f"Unknown extention '{ext}'.")
            lasers.append(
                Laser.from_structured(
                    data=data, config=config, name=name, filepath=path
                )
            )
    return lasers
