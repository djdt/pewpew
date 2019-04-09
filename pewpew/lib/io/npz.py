import os.path
import numpy as np

from pewpew import __version__

from typing import Any, Dict, List, Tuple
from pewpew.lib.laser import Laser, LaserConfig, LaserData
from pewpew.lib.exceptions import PewPewFileError


def load(path: str, config_override: LaserConfig = None) -> List[Laser]:
    """Imports the given numpy archive given, returning a list of data.

    Both the config and calibration read from the archive may be overriden.

    Args:
        path: Path to numpy archive
        config_override: If not None will be applied to all imports
        calibration_override: If not None will be applied to all imports

    Returns:
        List of LaserData and KrissKrossData

    Raises:
        PewPewFileError: Version of archive missing or incompatable.

    """
    lasers = []
    npz = np.load(path)

    if "version" not in npz.files:
        raise PewPewFileError("Archive version mismatch.")
    elif npz["version"] < "0.5.0":
        raise PewPewFileError(f"Archive version mismatch: {npz['version']}.")

    for i in range(0, npz["count"]):
        name = npz["name"][i]
        lasertype = npz["type"][i]
        config = (
            LaserConfig(**npz["config"][i])
            if config_override is None
            else config_override
        )
        data = {k: LaserData(**v) for k, v in npz["data"][i].items()}
        lasers.append(lasertype(data=data, config=config, name=name, filepath=path))

    return lasers


def save(path: str, laser_list: List[Laser]) -> None:
    savedict: Dict[str, Any] = {
        "version": __version__,
        "count": len(laser_list),
        "name": [],
        "type": [],
        "config": [],
        "data": [],
    }
    for laser in laser_list:
        savedict["name"].append(laser.name)
        savedict["type"].append(type(laser))
        savedict["config"].append(laser.config.__dict__)
        savedict["data"].append({k: v.__dict__ for k, v in laser.data.items()})
    np.savez_compressed(path, **savedict)
