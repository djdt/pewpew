from PySide2 import QtCore, QtHelp, QtWidgets
import logging
from pathlib import Path

from typing import Any


logger = logging.getLogger(__name__)


def createHelpEngine() -> QtHelp.QHelpEngine:
    cache = Path(
        QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.CacheLocation)
    )
    if not cache.exists():
        logger.info(f"Creating cache at '{cache}'.")
        cache.mkdir()
    engine = QtHelp.QHelpEngine(str(cache.joinpath("qthelp.qhc")))
    engine.setupData()

    qhc_path = str(Path(__file__).parent.joinpath("resources/pewpew.qch").absolute())
    namespace = QtHelp.QHelpEngineCore.namespaceName(qhc_path)

    if namespace not in engine.registeredDocumentations():
        logger.info("Registering help documentation.")
        if not engine.registerDocumentation(qhc_path):
            logger.info("Help registration failed!.")
            return None

    return engine


class HelpBrowser(QtWidgets.QTextBrowser):
    def __init__(self, engine: QtHelp.QHelpEngine, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setMinimumSize(1000, 800)
        self.engine = engine

    def loadResource(self, type: int, name: QtCore.QUrl) -> Any:
        if name.scheme() == "qthelp":
            return self.engine.fileData(name)
        return super().loadResource(type, name)


class HelpDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.engine = createHelpEngine()
        self.browser = HelpBrowser(self.engine)
        self.browser.setSource(QtCore.QUrl("qthelp://org.sphinx.pewpew/doc/index.html"))

        self.engine.contentWidget().linkActivated.connect(self.browser.setSource)
        self.engine.indexWidget().linkActivated.connect(self.browser.setSource)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.insertWidget(0, self.engine.contentWidget())
        splitter.insertWidget(1, self.browser)

        splitter.widget(0).setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)
