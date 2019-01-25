import os.path
import numpy as np

from pewpew.lib.laser import LaserData
from pewpew.lib.exceptions import PewPewDataError, PewPewFileError
from pewpew.lib.formatter import formatIsotope

from typing import Dict, List


def load(path: str, config: dict, calibration: dict = None) -> LaserData:
    """Imports iCap data exported using the CSV export function.

    Data is read from the "Counts" column.

    Args:
        path: Path to CSV
        config: Config to apply
        calibration: Calibration to apply

    Returns:
        The LaserData object.

    Raises:
        PewPewFileError: Unreadable file.
        PewPewDataError: Invalid data.

    """
    data: Dict[str, List[np.ndarray]] = {}
    with open(path, "r") as fp:
        # Find delimiter
        line = fp.readline().strip()
        delimiter = line[-1]
        # Skip row
        line = fp.readline()
        # First real row
        line = fp.readline()
        while line:
            try:
                _, _, isotope, data_type, line_data = line.split(delimiter, 4)
                if data_type == "Counter":
                    data.setdefault(formatIsotope(isotope), []).append(
                        np.genfromtxt(
                            [line_data],
                            delimiter=delimiter,
                            dtype=np.float64,
                            filling_values=0.0,
                        )
                    )
            except ValueError as e:
                raise PewPewFileError("Could not parse file.") from e
            line = fp.readline()

    keys = list(data.keys())
    # Stack lines to form 2d
    stacks: Dict[str, np.ndarray] = {}
    for k in keys:
        # Last line is junk
        stacks[k] = np.vstack(data[k])[:, :-1].transpose()
        if stacks[k].ndim != 2:
            raise PewPewDataError(f"Invalid data dimensions '{stacks[k].ndim}'.")

    # Build a named array out of data
    dtype = [(k, np.float64) for k in keys]
    shape = stacks[keys[0]].shape
    structured = np.empty(shape, dtype)
    for k in keys:
        if stacks[k].shape != shape:
            raise PewPewDataError("Mismatched data.")
        structured[k] = stacks[k]

    return LaserData(
        structured,
        config=config,
        calibration=calibration,
        name=os.path.splitext(os.path.basename(path))[0],
        source=path,
    )


def load_ldr(path: str, config: dict, calibration: dict = None) -> LaserData:
    """Imports data exported using \"Laser Data Reduction\".
    CSVs in the given directory are imported as
    lines in the image and are sorted by name.

    path -> path to directory containing CSVs
    config -> config to apply
    calibration -> calibration to apply

    returns LaserData"""
    data_files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.lower().endswith(".csv") and entry.is_file():
                data_files.append(entry.path)
    # Sort by name
    data_files.sort()

    with open(data_files[0], "r") as fp:
        line = fp.readline()
        skip_header = 0
        while line and not line.startswith("Time"):
            line = fp.readline()
            skip_header += 1

        delimiter = line[-1]

    cols = np.arange(1, line.count(delimiter))

    lines = [
        np.genfromtxt(
            f,
            delimiter=delimiter,
            names=True,
            usecols=cols,
            skip_header=skip_header,
            dtype=np.float64,
        )
        for f in data_files
    ]
    # We need to skip the first row as it contains junk
    data = np.vstack(lines)[1:]

    return LaserData(data, config=config, calibration=calibration, source=path)
