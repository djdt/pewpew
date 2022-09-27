from PySide2 import QtCore, QtGui
from pathlib import Path
import logging
import time

from pewpew import __version__ as pewpew_version
from pewlib import __version__ as pewlib_version
from pewlib import io
from pewlib import Config
from pewlib.laser import Laser

from typing import List, Optional


logger = logging.getLogger(__name__)


class ImportThread(QtCore.QThread):
    """Threaded file importer.

    To use connect importFinished and call 'run'.

    Args:
        paths: list of paths to import
        config: default config to apply

    Signals:
        importStarted: str, import started for path
        importFinished: Laser, laser object imported
        importFailed: str, import failed for path
        progressChanged: int, number of paths completed
    """

    importStarted = QtCore.Signal(str)
    importFinished = QtCore.Signal(object)
    importFailed = QtCore.Signal(str)
    progressChanged = QtCore.Signal(int)

    def __init__(
        self, paths: List[Path], config: Config, parent: Optional[QtCore.QObject] = None
    ):
        super().__init__(parent)
        self.paths = paths
        self.config = config

    def run(self) -> None:
        """Start the import thread."""
        for i, path in enumerate(self.paths):
            if self.isInterruptionRequested():  # pragma: no cover
                break
            self.progressChanged.emit(i)
            self.importStarted.emit(f"Importing {path.name}...")

            if QtGui.QImageReader.imageFormat(
                str(path.absolute())
            ):  # mime in QtGui.QImageReader.supportedMimeTypes():
                try:
                    image = QtGui.QImage(str(path))
                    self.importFinished.emit(image)
                except Exception as e:
                    logger.exception(e)
                    self.importFailed.emit(f"Unable to import {path.name}.")
                else:
                    logger.info(f"Imported image from {path.name}.")
            else:
                try:
                    laser = self.importPath(path)
                    self.importFinished.emit(laser)
                except Exception as e:
                    logger.exception(e)
                    self.importFailed.emit(f"Unable to import {path.name}.")
                else:
                    logger.info(f"Imported laser from {path.name}.")
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
        params = {}

        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found.")

        if path.is_dir():
            if path.suffix.lower() == ".b":
                data = None
                for methods in [["batch_xml", "batch_csv"], ["acq_method_xml"]]:
                    try:
                        data, params = io.agilent.load(
                            path, collection_methods=methods, full=True
                        )
                        info.update(io.agilent.load_info(path))
                        break
                    except ValueError as e:
                        logger.warning(f"Error for collection methods {methods}: {e}")
                if data is None:
                    raise ValueError(f"Unable to import batch '{path.name}'!")
            elif io.perkinelmer.is_valid_directory(path):
                data, params = io.perkinelmer.load(path, full=True)
                info["Instrument Vendor"] = "PerkinElemer"
            elif io.csv.is_valid_directory(path):
                data, params = io.csv.load(path, full=True)
            else:  # pragma: no cover
                raise ValueError(f"{path.name}: Unknown extention '{path.suffix}'.")
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
                    data = io.textimage.load(path, name="_element_")
            elif path.suffix.lower() in [".txt", ".text"]:
                data = io.textimage.load(path, name="_element_")
            else:  # pragma: no cover
                raise ValueError(f"{path.name}: Unknown extention '{path.suffix}'.")

        if "spotsize" in params:
            config.spotsize = params["spotsize"]
        if "speed" in params:
            config.speed = params["speed"]
        if "scantime" in params:
            config.scantime = params["scantime"]

        return Laser(data=data, config=config, info=info)
