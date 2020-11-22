import logging
from pathlib import Path

from pewlib import io
from pewlib import Laser, Config

from typing import List, Union

logger = logging.getLogger(__name__)

# PEW_VALID_EXTS = [".b", ".csv", ".npz", ".txt", ".text"]


def import_any(paths: List[Path], default_config: Config) -> List[Laser]:
    lasers: List[Laser] = []

    for path in paths:
        if not path.exists():
            continue

        config = Config(
            spotsize=default_config.spotsize,
            speed=default_config.speed,
            scantime=default_config.scantime,
        )

        logger.info(f"Importing {path.name}.")

        if path.is_dir():
            if path.suffix.lower() == ".b":
                data, params = io.agilent.load(path, full=True)
                config.scantime = params["scantime"]
            elif any([it.suffix.lower() == ".xl" for it in path.iterdir()]):
                data, params = io.perkinelmer.load(path, full=True)
                config.spotsize = params["spotsize"]
                config.speed = params["speed"]
                config.scantime = params["scantime"]
        else:
            if path.suffix.lower() == ".npz":
                lasers.append(io.npz.load(path))
                continue
            if path.suffix.lower() == ".csv":
                sample_format = io.thermo.icap_csv_sample_format(path)
                if sample_format in ["columns", "rows"]:
                    data, params = io.thermo.load(path, full=True)
                    config.scantime = params["scantime"]
                else:
                    data = io.textimage.load(path, name="_")
            elif path.suffix.lower() in [".txt", ".text"]:
                data = io.textimage.load(path, name="_")
            else:
                raise io.error.PewException(
                    f"{path.name}: Unknown extention '{path.suffix}'."
                )

        lasers.append(
            Laser(data=data, config=config, name=path.stem, path=path.resolve())
        )
    return lasers
