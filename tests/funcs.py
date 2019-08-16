from PySide2 import QtCore, QtWidgets


def check_top_level_widgets(type: type):
    widgets = QtWidgets.QApplication.topLevelWidgets()
    # print([type(w) for w in widgets])
    for w in widgets:
        if isinstance(w, type):
            return w
    return None


def wait_for_and_close_top_level(type: type, wait_time: int = 10, max_time: int = 1000):
    if max_time < 0:
        QtWidgets.QApplication.exit()
        raise TimeoutError

    # w = QtWidgets.QApplication.activeModalWidget()
    w = check_top_level_widgets(type)
    if w is not None:
        w.close()  # type: ignore
    else:
        QtCore.QTimer.singleShot(
            wait_time,
            lambda: wait_for_and_close_top_level(type, wait_time, max_time - wait_time),
        )


def wait_for_and_close_modal(
    type: type, wait_time: int = 10, max_time: int = 1000
) -> None:
    if max_time < 0:
        QtWidgets.QApplication.exit()
        raise TimeoutError

    w = QtWidgets.QApplication.activeModalWidget()
    if isinstance(w, type):
        w.close()
    else:
        QtCore.QTimer.singleShot(
            wait_time,
            lambda: wait_for_and_close_modal(type, wait_time, max_time - wait_time),
        )
