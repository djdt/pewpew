import os.path

from laserlib import io
from laserlib import Laser, LaserConfig

from typing import List


def import_thermo_csv(path: str, config: LaserConfig) -> Laser:
    base, ext = os.path.splitext(path)
    name = os.path.basename(base)
    ext = ext.lower()
    if ext.lower() != ".csv":
        raise io.error.LaserLibException(
            f"{name+ext}: Invalid extention '{ext}' for iCap CSV."
        )
    data = io.thermo.load(path)
    return Laser.from_structured(data=data, config=config, name=name, filepath=path)


def import_agilent_batch(path: str, config: LaserConfig) -> Laser:
    base, ext = os.path.splitext(path)
    name = os.path.basename(base)
    ext = ext.lower()
    if ext.lower() != ".b":
        raise io.error.LaserLibException(
            f"{name+ext}: Invalid extention '{ext}' for Agilent batch."
        )
    data = io.agilent.load(path)
    return Laser.from_structured(data=data, config=config, name=name, filepath=path)


def import_csv(path: str, config: LaserConfig) -> Laser:
    base, ext = os.path.splitext(path)
    name = os.path.basename(base)
    ext = ext.lower()
    if ext.lower() not in [".csv", ".txt", ".text"]:
        raise io.error.LaserLibException(
            f"{name+ext}: Invalid extention '{ext}' for CSV."
        )
    data = io.csv.load(path)
    return Laser.from_structured(data=data, config=config, name=name, filepath=path)


def import_any(paths: List[str], config: LaserConfig) -> List[Laser]:
    lasers = []
    for path in paths:
        _, ext = os.path.splitext(path.lower())
        if ext == ".npz":
            lasers.extend(io.npz.load(path))
        elif ext == ".csv":
            try:
                lasers.append(import_thermo_csv(path, config))
            except io.error.LaserLibException:
                lasers.append(import_csv(path, config))
        elif ext in [".txt", ".text"]:
            lasers.append(import_csv(path, config))
        elif ext == ".b":
            lasers.append(import_agilent_batch(path, config))
        else:
            raise io.error.LaserLibException(f"Unknown extention '{ext}'.")
    return lasers
