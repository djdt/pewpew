from pathlib import Path

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.actions import qAction


class LimitBoundViewBox(pyqtgraph.ViewBox):
    """Viewbox that autoRanges to any set limits."""

    def childrenBounds(self, frac=None, orthoRange=(None, None), items=None):
        bounds = super().childrenBounds(frac=frac, orthoRange=orthoRange, items=items)
        limits = self.state["limits"]["xLimits"], self.state["limits"]["yLimits"]
        for i in range(2):
            if bounds[i] is not None:
                if limits[i][0] != -1e307:  # and limits[i][0] < bounds[i][0]:
                    bounds[i][0] = limits[i][0]
                if limits[i][1] != +1e307:  # and limits[i][1] > bounds[i][1]:
                    bounds[i][1] = limits[i][1]
        return bounds


class ViewBoxForceScaleAtZero(LimitBoundViewBox):
    """Viewbox that forces the bottom to be 0."""

    def scaleBy(
        self,
        s: list[float] | None = None,
        center: QtCore.QPointF | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        if center is not None:
            center.setY(0.0)
        super().scaleBy(s, center, x, y)

    def translateBy(
        self,
        t: QtCore.QPointF | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        if t is not None:
            t.setY(0.0)
        if y is not None:
            y = 0.0
        super().translateBy(t, x, y)


class SinglePlotGraphicsView(pyqtgraph.GraphicsView):
    def __init__(
        self,
        title: str,
        xlabel: str = "",
        ylabel: str = "",
        xunits: str | None = None,
        yunits: str | None = None,
        viewbox: pyqtgraph.ViewBox | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(background="white", parent=parent)

        pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        pen.setCosmetic(True)

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel(xlabel, units=xunits)

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel(ylabel, units=yunits)

        self.plot = pyqtgraph.PlotItem(
            title=title,
            name="central_plot",
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            viewBox=viewbox,
        )
        # Common options
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.addLegend(
            offset=(-5, 5), verSpacing=-5, colCount=1, labelTextColor="black"
        )

        self.setCentralWidget(self.plot)

        self.action_copy_image = qAction(
            "insert-image",
            "Copy Image to Clipboard",
            "Copy an image of the plot to the clipboard.",
            self.copyToClipboard,
        )

        # self.action_show_legend = qAction(
        #     "view-hidden",
        #     "Hide Legend",
        #     "Toggle visibility of the legend.",
        #     lambda: None,
        # )

        self.action_export_data = qAction(
            "document-export",
            "Export Data",
            "Save currently loaded data to file.",
            self.exportData,
        )

        self.context_menu_actions: list[QtGui.QAction] = []

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_copy_image)
        # if self.plot.legend is not None:
        #     if self.plot.legend.isVisible():
        #         self.action_show_legend.setIcon(QtGui.QIcon.fromTheme("view-hidden"))
        #         self.action_show_legend.setText("Hide Legend")
        #         self.action_show_legend.triggered.connect(
        #             lambda: self.plot.legend.setVisible(False)
        #         )
        #     else:
        #         self.action_show_legend.setIcon(QtGui.QIcon.fromTheme("view-visible"))
        #         self.action_show_legend.setText("Show Legend")
        #         self.action_show_legend.triggered.connect(
        #             lambda: self.plot.legend.setVisible(True)
        #         )
        #
        #     menu.addAction(self.action_show_legend)
        if self.readyForExport():
            menu.addSeparator()
            menu.addAction(self.action_export_data)

        if len(self.context_menu_actions) > 0:
            menu.addSeparator()
        for action in self.context_menu_actions:
            menu.addAction(action)

        event.accept()
        menu.popup(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

    def clear(self) -> None:
        self.plot.legend.clear()
        self.plot.clear()

    def dataBounds(self) -> tuple[float, float, float, float]:
        items = [item for item in self.plot.listDataItems() if item.isVisible()]
        bx = np.array([item.dataBounds(0) for item in items], dtype=float)
        by = np.array([item.dataBounds(1) for item in items], dtype=float)
        bx, by = np.nan_to_num(bx), np.nan_to_num(by)
        # Just in case
        if len(bx) == 0 or len(by) == 0:
            return (0, 1, 0, 1)
        return (
            np.amin(bx[:, 0]),
            np.amax(bx[:, 1]),
            np.amin(by[:, 0]),
            np.amax(by[:, 1]),
        )

    def dataRect(self) -> QtCore.QRectF:
        x0, x1, y0, y1 = self.dataBounds()
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

    def exportData(self) -> None:
        dir = QtCore.QSettings().value("RecentFiles/1/path", None)
        dir = str(Path(dir).parent) if dir is not None else ""
        path, filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Data",
            dir,
            "CSV Documents(*.csv);;Numpy archives(*.npz);;All Files(*)",
        )
        if path == "":
            return

        path = Path(path)

        filter_suffix = filter[filter.rfind(".") : -1]
        if filter_suffix != "":  # append suffix if missing
            path = path.with_suffix(filter_suffix)

        data = self.dataForExport()
        names = list(data.keys())

        if path.suffix.lower() == ".csv":
            header = "\t".join(name for name in names)
            stack = np.full(
                (max(d.size for d in data.values()), len(data)),
                np.nan,
                dtype=np.float32,
            )
            for i, x in enumerate(data.values()):
                stack[: x.size, i] = x
            np.savetxt(
                path, stack, delimiter="\t", comments="", header=header, fmt="%.16g"
            )
        elif path.suffix.lower() == ".npz":
            np.savez_compressed(
                path,
                **{k: v for k, v in data.items()},
            )
        else:
            raise ValueError("dialogExportData: file suffix must be '.npz' or '.csv'.")

    def dataForExport(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def readyForExport(self) -> bool:
        return False

    def setLimits(self, **kwargs) -> None:
        self.plot.setLimits(**kwargs)

    def setDataLimits(
        self,
        xMin: float | None = None,
        xMax: float | None = None,
        yMin: float | None = None,
        yMax: float | None = None,
    ) -> None:
        """Set all plots limits in range 0.0 - 1.0."""
        bounds = self.dataBounds()
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        limits = {}
        if xMin is not None:
            limits["xMin"] = bounds[0] + dx * xMin
        if xMax is not None:
            limits["xMax"] = bounds[0] + dx * xMax
        if yMin is not None:
            limits["yMin"] = bounds[2] + dy * yMin
        if yMax is not None:
            limits["yMax"] = bounds[2] + dy * yMax
        self.setLimits(**limits)

    def zoomReset(self) -> None:
        x, y = self.plot.vb.state["autoRange"][0], self.plot.vb.state["autoRange"][1]
        self.plot.autoRange()
        self.plot.enableAutoRange(x=x, y=y)

        # Reset the legend postion
        if self.plot.legend is not None:
            self.plot.legend.anchor(
                QtCore.QPointF(1, 0), QtCore.QPointF(1, 0), QtCore.QPointF(-10, 10)
            )
