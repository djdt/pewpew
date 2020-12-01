from pytestqt.qtbot import QtBot
from pathlib import Path

from pewlib.config import Config

from pewpew.threads import ImportThread


def test_import_thread(qtbot: QtBot):
    path = Path(__file__).parent.joinpath("data", "io")
    paths = [
        path.joinpath("agilent", "test_ms.b"),
        path.joinpath("npz", "test.npz"),
        path.joinpath("perkinelmer", "perkinelmer"),
        path.joinpath("textimage", "csv.csv"),
        path.joinpath("thermo", "icap_columns.csv"),
    ]
    thread = ImportThread(paths, Config())

    signals = [thread.importFinished] * len(paths)
    signals.extend([thread.progressChanged] * len(paths))
    with qtbot.waitSignals(signals):
        thread.run()

    # Failing import
    paths = [path.joinpath("fake", "data.npz")]
    thread = ImportThread(paths, Config())

    with qtbot.waitSignal(thread.importFailed):
        thread.run()
