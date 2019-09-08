from PySide2 import QtCore, QtGui, QtWidgets

from typing import Callable


class ActionGenerator(object):
    ACTIONS = {
        "copy-image": ("Copy &Image", "insert-image", "Copy image to clipboard."),
        "open": ("&Open", "document-open", "Open a new document from a numpy archive."),
        # "import": ("Import", "document-open", "Open a new document from a numpy archive."),
        "save": ("&Save", "document-save", "Save document to numpy archive."),
        "export": (
            "E&xport",
            "document-save-as",
            "Export document to selected format.",
        ),
        "close": ("Close", "edit-delete", "Close document."),
        "quit": ("&Quit", "application-exit", "Quit the application"),
        "about": ("About", "help-about", "About this program."),
        "dialog-calibration": (
            "Ca&libration",
            "go-top",
            "Open the calibration dialog.",
        ),
        "dialog-config": ("Config", "document-edit", "Open the configuration dialog."),
        "dialog-statistics": (
            "&Statistics",
            "",
            "Open the histogram and statistics dialog.",
        ),
    }

    @staticmethod
    def getAction(
        name: str,
        function: Callable,
        label: str = None,
        icon: str = None,
        status: str = None,
    ) -> QtWidgets.QAction:
        if name in ActionGenerator.ACTIONS:
            label = ActionGenerator.ACTIONS[name][0] if label is None else label
            icon = ActionGenerator.ACTIONS[name][1] if icon is None else icon
            status = ActionGenerator.ACTIONS[name][2] if status is None else status

        action = QtWidgets.QAction(QtGui.QIcon.fromTheme(icon), label)
        action.setStatusTip(status)
        action.triggered.connect(function)
        return action
