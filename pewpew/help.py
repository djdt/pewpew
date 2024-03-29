import logging
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtHelp, QtWidgets

logger = logging.getLogger(__name__)


def createHelpEngine() -> QtHelp.QHelpEngine:
    """Create the pewpew help engine."""
    cache = Path(
        QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.CacheLocation)
    )
    if not cache.exists():
        logger.info(f"Creating cache at '{cache}'.")
        cache.mkdir(parents=True)

    qch_path = Path(__file__).parent.joinpath("resources/pewpew.qch").absolute()
    if not qch_path.exists():
        logger.warning(f"Help file '{qch_path}' does not exist.")

    engine = QtHelp.QHelpEngine(str(cache.joinpath("pewpew_help.qhc")))
    engine.setReadOnly(False)  # see https://bugreports.qt.io/browse/QTBUG-106028
    engine.setupData()

    namespace = engine.namespaceName(str(qch_path))
    if namespace not in engine.registeredDocumentations():
        logger.info("Registering help documentation.")
        if not engine.registerDocumentation(str(qch_path)):
            logger.warning(
                f"Help registration failed, '{engine.error()}'. Is readonly?"
            )

    return engine


class HelpBrowser(QtWidgets.QTextBrowser):
    """Text browser for a QHelpEngine."""

    def __init__(
        self, engine: QtHelp.QHelpEngine, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setMinimumSize(1000, 800)
        self.engine = engine

    def loadResource(self, type: int, name: QtCore.QUrl) -> Any:
        if name.scheme() == "qthelp":
            return self.engine.fileData(name)
        return super().loadResource(type, name)


class HelpDialog(QtWidgets.QDialog):
    """Dialog to display the pewpew help files."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("pewpew Help")

        self.engine = createHelpEngine()
        self.engine.searchEngine().reindexDocumentation()

        self.browser = HelpBrowser(self.engine)
        self.browser.setSource(QtCore.QUrl("qthelp://org.sphinx.pewpew/doc/index.html"))

        self.engine.contentWidget().linkActivated.connect(self.browser.setSource)
        self.engine.indexWidget().linkActivated.connect(self.browser.setSource)
        self.engine.searchEngine().resultWidget().requestShowLink.connect(
            self.browser.setSource
        )
        self.engine.searchEngine().queryWidget().search.connect(self.search)

        self.button_back = QtWidgets.QToolButton()
        self.button_back.setIcon(QtGui.QIcon.fromTheme("arrow-left"))
        self.button_back.pressed.connect(self.browser.backward)

        self.button_forward = QtWidgets.QToolButton()
        self.button_forward.setIcon(QtGui.QIcon.fromTheme("arrow-right"))
        self.button_forward.pressed.connect(self.browser.forward)

        search = QtWidgets.QWidget()
        search.setLayout(QtWidgets.QVBoxLayout())
        search.layout().addWidget(self.engine.searchEngine().queryWidget(), 0)
        search.layout().addWidget(self.engine.searchEngine().resultWidget(), 1)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.engine.contentWidget(), "Content")
        tabs.addTab(self.engine.indexWidget(), "Index")
        tabs.addTab(search, "Search")
        tabs.setCurrentIndex(1)

        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addStretch(1)
        layout_buttons.addWidget(self.button_back, 0, QtCore.Qt.AlignRight)
        layout_buttons.addWidget(self.button_forward, 0, QtCore.Qt.AlignRight)

        container = QtWidgets.QWidget()
        container.setLayout(QtWidgets.QVBoxLayout())
        container.layout().addLayout(layout_buttons, 0)
        container.layout().addWidget(self.browser, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.insertWidget(0, tabs)
        splitter.insertWidget(1, container)

        splitter.widget(0).setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

    def search(self) -> None:
        input = self.engine.searchEngine().queryWidget().searchInput()
        self.engine.searchEngine().search(input)
