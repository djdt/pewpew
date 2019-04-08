import os.path
import numpy as np

from pewpew import __version__

from typing import Any, Dict, List
from pewpew.lib.laser import Laser, LaserConfig
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
    lds = []
    npz = np.load(path)

    if "version" not in npz.files:
        raise PewPewFileError("Archive version mismatch.")
    elif npz["version"] < "0.5.0":
        raise PewPewFileError(f"Archive version mismatch: {npz['version']}.")

    num_files = sum(1 for d in npz.files if "_data" in d)
    for i in range(0, num_files):
        name = (
            npz["name"][i]
            if "name" in npz.files
            else os.path.splitext(os.path.basename(path))[0]
        )
        type = npz["type"][i] if "type" in npz.files else Laser
        if config_override is None:
            config = npz["config"][i] if "config" in npz.files else None
        else:
            config = config_override
        ld = type(config=config, name=name, filepath=path)
        for key in npz.files:

    return lds


def _config_to_array(config: LaserConfig) -> Tuple[float, float, float]:
    return config.scantime, config.speed, config.spotsize


def _data_to_array(data: LaserData) -> Tuple[np.ndarray, float, float, str]:
    return data.data, data.intercept, data.gradient, data.unit


def save(path: str, laser_list: List[Laser]) -> None:
    savedict: Dict[str, Any] = {
        "version": __version__,
        "name": [],
        "type": [],
        "config": [],
    }
    for i, laser in enumerate(laser_list):
        savedict["name"].append(laser.name)
        savedict["type"].append(type(laser))
        savedict["config"].append(_config_to_array(laser.config))
        for name in laser.names():
            savedict[f"data_{i}_{name}"] = _data_to_array(laser.data)
    np.savez_compressed(path, **savedict)


if __name__ == "__main__":

