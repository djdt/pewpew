from PySide6 import QtCore, QtGui, QtWidgets

from typing import Callable


def qAction(icon: str, label: str, status: str, func: Callable) -> QtGui.QAction:
    """Create a QAction.

    Args:
        icon: passed to QIcon.fromTheme
        label: action label
        status: status- and tooltip
        func: connected to triggered
    """
    action = QtGui.QAction(QtGui.QIcon.fromTheme(icon), label)
    action.setStatusTip(status)
    action.setToolTip(status)
    action.triggered.connect(func)
    return action


def qActionGroup(
    parent: QtWidgets.QWidget,
    actions: list[str],
    func: Callable,
    statuses: list[str] | None = None,
    icons: list[QtGui.QIcon] | None = None,
    checked: str | None = None,
) -> QtGui.QActionGroup:
    """Create a QActionGroup.

    Args:
        actions: list of action labels
        func: connected to group triggered
        statuses: list of status tips
        checked: check action with this label
    """
    group = QtGui.QActionGroup(parent)
    for i, name in enumerate(actions):
        action = group.addAction(name)
        if statuses is not None:
            action.setStatusTip(statuses[i])
        if icons is not None:
            action.setIcon(icons[i])
        if checked is not None:
            action.setCheckable(True)
            if name == checked:
                action.setChecked(True)
    group.triggered.connect(func)

    return group


def qToolButton(
    icon: str | None = None,
    text: str | None = None,
    action: QtGui.QAction | None = None,
    parent: QtWidgets.QWidget | None = None,
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
