from PySide2 import QtCore, QtGui, QtWidgets

from typing import Callable, List


def qAction(icon: str, label: str, status: str, func: Callable) -> QtWidgets.QAction:
    """Create a QAction.

    Args:
        icon: passed to QIcon.fromTheme
        label: action label
        status: status- and tooltip
        func: connected to triggered
    """
    action = QtWidgets.QAction(QtGui.QIcon.fromTheme(icon), label)
    action.setStatusTip(status)
    action.setToolTip(status)
    action.triggered.connect(func)
    return action


def qActionGroup(
    parent: QtWidgets.QWidget,
    actions: List[str],
    func: Callable,
    statuses: List[str] = None,
    checked: str = None,
) -> QtWidgets.QActionGroup:
    """Create a QActionGroup.

    Args:
        actions: list of action labels
        func: connected to group triggered
        statuses: list of status tips
        checked: check action with this label
    """
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


def qToolButton(
    icon: str = None,
    text: str = None,
    action: QtWidgets.QAction = None,
    parent: QtWidgets.QWidget = None,
) -> QtWidgets.QToolButton:
    """Create a styled QToolButton.
    If 'action' is provided then the actions 'icon and 'text are displayed,
    otherwise displays 'icon' and 'text'.

    Args:
        icon: passed to QIcon.fromTheme
        text: button text
        action: buttons action
        parent: parent of button
    """
    button = QtWidgets.QToolButton(parent)
    button.setAutoRaise(True)
    button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
    if icon is not None:
        button.setIcon(QtGui.QIcon.fromTheme(icon))
    if text is not None:
        button.setText(text)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
    if action is not None:
        button.setDefaultAction(action)

    return button
