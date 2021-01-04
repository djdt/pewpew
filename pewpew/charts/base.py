from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts


class BaseChart(QtCharts.QChartView):
    def contextMenuEvent(self, event: QtCore.QEvent) -> None:
        action_copy_image = QtWidgets.QAction(
            QtGui.QIcon.fromTheme("insert-image"), "Copy To Clipboard", self
        )
        action_copy_image.setStatusTip("Copy the graphics view to the clipboard.")
        action_copy_image.triggered.connect(self.copyToClipboard)

        context_menu = QtWidgets.QMenu(self.parent())
        context_menu.addAction(action_copy_image)
        context_menu.popup(event.globalPos())

    def copyToClipboard(self) -> None:
        QtWidgets.QApplication.clipboard().setPixmap(self.grab(self.viewport().rect()))
