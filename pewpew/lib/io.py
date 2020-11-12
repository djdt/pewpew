import logging
import os.path

from pew import io
from pew import Laser, Config

from typing import List

logger = logging.getLogger(__name__)

PEW_VALID_EXTS = [".b", ".csv", ".npz", ".txt", ".text"]


def import_any(paths: List[str], config: Config) -> List[Laser]:
    lasers: List[Laser] = []
    for path in paths:
        base, ext = os.path.splitext(path)
        name = os.path.basename(base)
        ext = ext.lower()

        logger.info(f"Importing {name}.")

        if ext == ".npz":
            lasers.append(io.npz.load(path))
        else:
            if ext == ".csv":
                try:
                    data, params = io.thermo.load(path, full=True)
                    config.scantime = params["scantime"]
                except io.error.PewException:
                    data = io.csv.load(path, isotope="_")
            elif ext in [".txt", ".text"]:
                data = io.csv.load(path, isotope="_")
            elif ext == ".b":
                data, params = io.agilent.load(path, full=True)
                config.scantime = params["scantime"]
            else:
                raise io.error.PewException(f"{name}: Unknown extention '{ext}'.")
            lasers.append(Laser(data=data, config=config, name=name, path=path))
    return lasers
