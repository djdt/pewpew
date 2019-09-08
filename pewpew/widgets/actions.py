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

    def __init__(self):

        self.action_copy_image = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy Image", self
        )
        self.action_copy_image.setStatusTip("Copy image to clipboard.")
        self.action_copy_image.triggered.connect(self.canvas.copyToClipboard)

        self.action_save = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("document-save"), "Save", self
        )
        self.action_save.setStatusTip("Save data to archive.")
        self.action_save.triggered.connect(self.onMenuSave)

        self.action_export = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("document-save-as"), "Export", self
        )
        self.action_export.setStatusTip("Save data to different formats.")
        self.action_export.triggered.connect(self.onMenuExport)

        self.action_calibration = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("go-top"), "Calibration", self
        )
        self.action_calibration.setStatusTip("Edit image calibration.")
        self.action_calibration.triggered.connect(self.onMenuCalibration)

        self.action_config = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("document-edit"), "Config", self
        )
        self.action_config.setStatusTip("Edit image config.")
        self.action_config.triggered.connect(self.onMenuConfig)

        self.action_stats = QtWidgets.QAction(
            QtGui.QIcon.fromTheme(""), "Statistics", self
        )
        self.action_stats.setStatusTip("Data histogram and statistics.")
        self.action_stats.triggered.connect(self.onMenuStats)

        self.action_close = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("edit-delete"), "Close", self
        )
        self.action_close.setStatusTip("Close the images.")
        self.action_close.triggered.connect(self.onMenuClose)
