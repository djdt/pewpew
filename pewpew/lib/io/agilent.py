import os.path
import numpy as np

from pewpew.lib.laser import LaserData
from pewpew.lib.exceptions import PewPewDataError, PewPewFileError
from pewpew.lib.formatter import formatIsotope


def load(path: str, config: dict, calibration: dict = None) -> LaserData:
    """Imports an Agilent batch (.b) directory, returning LaserData object.

   Scans the given path for .d directories containg a similarly named
   .csv file. These are imported as lines, sorted by their name.

    Args:
       path: Path to the .b directory
       config: Config to be applied
       calibration: Calibration to be applied

    Returns:
        The LaserData object.

    Raises:
        PewPewFileError: Missing or malformed files.
        PewPewDataError: Invalid data.

    """
    data_files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.lower().endswith(".d") and entry.is_dir():
                csv = entry.name[: entry.name.rfind(".")] + ".csv"
                csv = os.path.join(entry.path, csv)
                if not os.path.exists(csv):
                    raise PewPewFileError(f"Missing csv '{csv}'.")
                data_files.append(csv)
    # Sort by name
    data_files.sort()

    with open(data_files[0], "r") as fp:
        line = fp.readline()
        skip_header = 0
        while line and not line.startswith("Time [Sec]"):
            line = fp.readline()
            skip_header += 1

        skip_footer = 0
        if "Print" in fp.read().splitlines()[-1]:
            skip_footer = 1

    cols = np.arange(1, line.count(",") + 1)

    try:
        lines = [
            np.genfromtxt(
                f,
                delimiter=",",
                names=True,
                usecols=cols,
                skip_header=skip_header,
                skip_footer=skip_footer,
                dtype=np.float64,
            )
            for f in data_files
        ]
    except ValueError as e:
        raise PewPewFileError("Could not parse batch.") from e

    try:
        data = np.vstack(lines)
    except ValueError as e:
        raise PewPewDataError("Mismatched data.") from e

    # Format isotope names
    data.dtype.names = [formatIsotope(name) for name in data.dtype.names]

    return LaserData(
        data,
        config=config,
        calibration=calibration,
        name=os.path.splitext(os.path.basename(path))[0],
        source=path,
    )
