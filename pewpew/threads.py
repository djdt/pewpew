from PySide2 import QtCore
from pathlib import Path
import logging
import time

from pewpew import __version__ as pewpew_version
from pewlib import __version__ as pewlib_version
from pewlib import io
from pewlib import Config, Laser

from typing import List


logger = logging.getLogger(__name__)


class ImportThread(QtCore.QThread):
    importStarted = QtCore.Signal(str)
    importFinished = QtCore.Signal(object)
    importFailed = QtCore.Signal(str)
    progressChanged = QtCore.Signal(int)

    def __init__(
        self, paths: List[Path], config: Config, parent: QtCore.QObject = None
    ):
        super().__init__(parent)
        self.paths = paths
        self.config = config

    def run(self) -> None:
        for i, path in enumerate(self.paths):
            if self.isInterruptionRequested():  # pragma: no cover
                break
            self.progressChanged.emit(i)
            self.importStarted.emit(f"Importing {path.name}...")
            try:
                laser = self.importPath(path)
                self.importFinished.emit(laser)
                logger.info(f"Imported {path.name}.")
            except Exception as e:
                logger.exception(e)
                self.importFailed.emit(f"Unable to import {path.name}.")
        self.progressChanged.emit(len(self.paths))

    def importPath(self, path: Path) -> Laser:
        config = Config(
            spotsize=self.config.spotsize,
            speed=self.config.speed,
            scantime=self.config.scantime,
        )
        info = {
            "Name": path.stem,
            "File Path": str(path.resolve()),
            "Import Date": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time())
            ),
            "Import Path": str(path.resolve()),
            "Import Version pewlib": pewlib_version,
            "Import Version pew2": pewpew_version,
        }

        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found.")

        if path.is_dir():
            if path.suffix.lower() == ".b":
                data, params = io.agilent.load(path, full=True)
                info.update(io.agilent.load_info(path))
            elif io.perkinelmer.is_valid_directory(path):
                data, params = io.perkinelmer.load(path, full=True)
                info["Instrument Vendor"] = "PerkinElemer"
            elif io.csv.is_valid_directory(path):
                data, params = io.csv.load(path, full=True)
        else:
            if path.suffix.lower() == ".npz":
                laser = io.npz.load(path)
                return laser
            if path.suffix.lower() == ".csv":
                sample_format = io.thermo.icap_csv_sample_format(path)
                if sample_format in ["columns", "rows"]:
                    data, params = io.thermo.load(path, full=True)
                    info["Instrument Vendor"] = "Thermo"
                else:
                    data = io.textimage.load(path, name="_isotope_")
            elif path.suffix.lower() in [".txt", ".text"]:
                data = io.textimage.load(path, name="_isotope_")
            else:  # pragma: no cover
                raise ValueError(f"{path.name}: Unknown extention '{path.suffix}'.")

        if "spotsize" in params:
            config.spotsize = params["spotsize"]
        if "speed" in params:
            config.speed = params["speed"]
        if "scantime" in params:
            config.scantime = params["scantime"]

        return Laser(data=data, config=config, info=info)
