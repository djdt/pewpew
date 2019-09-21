from PySide2 import QtWidgets, QtGui

from typing import Callable, List


def qAction(icon: str, label: str, status: str, func: Callable) -> QtWidgets.QAction:
    action = QtWidgets.QAction(QtGui.QIcon.fromTheme(icon), label)
    action.setStatusTip(status)
    action.triggered.connect(func)
    return action


def qActionGroup(
    parent: QtWidgets.QWidget,
    actions: List[str],
    func: Callable,
    statuses: List[str] = None,
    checked: str = None,
) -> QtWidgets.QActionGroup:
    group = QtWidgets.QActionGroup(parent)
    for i, name in enumerate(actions):
        action = group.addAction(name)
        if statuses is not None:
            action.setStatusTip(statuses[i])
        if checked is not None:
            action.setCheckable(True)
            if name == checked:
                action.setChecked(True)
    group.triggered.connect(func)

    return group
