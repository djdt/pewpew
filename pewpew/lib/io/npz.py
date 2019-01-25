import os.path
import numpy as np

from pewpew import __version__

from typing import Any, Dict, List
from pewpew.lib.laser import LaserData
from pewpew.lib.exceptions import PewPewFileError


def load(
    path: str, config_override: dict = None, calibration_override: dict = None
) -> List[LaserData]:
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
    elif npz["version"] < "0.3.3":
        raise PewPewFileError(f"Archive version mismatch: {npz['version']}.")

    num_files = sum(1 for d in npz.files if "_data" in d)
    for i in range(0, num_files):
        name = (
            npz["_name"][i]
            if "_name" in npz.files
            else os.path.splitext(os.path.basename(path))[0]
        )
        type = npz["_type"][i] if "_type" in npz.files else LaserData
        if config_override is None:
            config = npz["_config"][i] if "_config" in npz.files else None
        else:
            config = config_override
        if calibration_override is None:
            calibration = (
                npz["_calibration"][i] if "_calibration" in npz.files else None
            )
        else:
            calibration = calibration_override
        lds.append(
            type(
                data=npz[f"_data{i}"],
                config=config,
                calibration=calibration,
                name=name,
                source=path,
            )
        )
    return lds


def save(path: str, laser_list: List[LaserData]) -> None:
    savedict: Dict[str, Any] = {
        "version": __version__,
        "_name": [],
        "_type": [],
        "_config": [],
        "_calibration": [],
    }
    for i, laser in enumerate(laser_list):
        savedict["_name"].append(laser.name)
        savedict["_type"].append(type(laser))
        savedict["_config"].append(laser.config)
        savedict["_calibration"].append(laser.calibration)
        savedict[f"_data{i}"] = laser.data
    np.savez_compressed(path, **savedict)
