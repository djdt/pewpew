import numpy as np
from pewlib.process.calc import normalise
from pewlib.process.threshold import otsu
from PySide6 import QtCore, QtGui, QtWidgets

from pewpew.graphics import colortable
from pewpew.graphics.imageitems import LaserImageItem, ScaledImageItem
from pewpew.lib import kmeans
from pewpew.lib.pratt import (
    BinaryFunction,
    Parser,
    ParserException,
    Reducer,
    ReducerException,
    TernaryFunction,
    UnaryFunction,
)
from pewpew.widgets.ext import ValidColorLineEdit, ValidColorTextEdit
from pewpew.widgets.tools import ToolWidget
from pewpew.widgets.views import TabView


def segment_image(x: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    mask = np.zeros(x.shape, dtype=int)
    for i, t in enumerate(np.atleast_1d(thresholds)):
        mask[x > t] = i + 1
    return mask


class CalculatorName(ValidColorLineEdit):
    """A lineedit that excludes invalid or existing element names.

    Colors red on bad input.
    """

    def __init__(
        self,
        text: str,
        badnames: list[str],
        badparser: list[str],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(text, parent=parent)

        self.badnames = badnames
        self.badchars = [" ", "\t", "\n"]
        self.badnulls = ["nan", "if", "then", "else"]
        self.badnulls.extend(badparser)

    def hasAcceptableInput(self) -> bool:
        if self.text() == "":
            return False
        if any(char in self.text() for char in self.badchars):
            return False
        if self.text() in self.badnulls:
            return False
        if self.text() in self.badnames:
            return False
        return True


class CalculatorFormula(ValidColorTextEdit):
    """Input for the calculator.

    Parsers input using a `:class:pewpew.lib.pratt.Parser` and
    colors input red when invalid. Implements completion when `completer` is set.
    """

    def __init__(
        self,
        text: str,
        variables: list[str],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(text, parent)

        self.completer: QtWidgets.QCompleter | None = None

        self.textChanged.disconnect(self.revalidate)
        self.textChanged.connect(self.calculate)
        self.parser = Parser(variables)
        self.expr = ""

    def hasAcceptableInput(self) -> bool:
        return self.expr != ""

    def calculate(self) -> None:
        try:
            self.expr = self.parser.parse(self.toPlainText())
        except ParserException:
            self.expr = ""
        self.revalidate()

    def setCompleter(self, completer: QtWidgets.QCompleter) -> None:
        """Set the completer used."""
        if self.completer is not None:
            self.completer.disconnect(self)

        self.completer = completer
        self.completer.setWidget(self)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.completer.activated.connect(self.insertCompletion)

    def insertCompletion(self, completion: str) -> None:
        tc = self.textCursor()
        for _ in range(len(self.completer.completionPrefix())):
            tc.deletePreviousChar()
        tc.insertText(completion)
        self.setTextCursor(tc)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.completer is not None and self.completer.popup().isVisible():
            if event.key() in [  # Ignore keys when popup is present
                QtCore.Qt.Key_Enter,
                QtCore.Qt.Key_Return,
                QtCore.Qt.Key_Escape,
                QtCore.Qt.Key_Tab,
                QtCore.Qt.Key_Down,
                QtCore.Qt.Key_Up,
            ]:
                event.ignore()
                return

        super().keyPressEvent(event)

        eow = "~!@#$%^&*()+{}|:\"<>?,./;'[]\\-="
        tc = self.textCursor()
        tc.select(QtGui.QTextCursor.WordUnderCursor)
        prefix = tc.selectedText()
        if prefix != self.completer.completionPrefix():
            self.completer.setCompletionPrefix(prefix)
            self.completer.popup().setCurrentIndex(
                self.completer.completionModel().index(0, 0)
            )

        if (
            len(prefix) < 2
            or event.text() == ""
            or event.text()[-1] in eow
            or prefix == self.completer.currentCompletion()
        ):
            self.completer.popup().hide()
        else:
            rect = self.cursorRect()
            rect.setWidth(
                self.completer.popup().sizeHintForColumn(0)
                + self.completer.popup().verticalScrollBar().sizeHint().width()
            )
            self.completer.complete(rect)


class CalculatorTool(ToolWidget):
    """Calculator for element data operations."""

    functions = {
        "abs": (
            (UnaryFunction("abs"), "(<x>)", "The absolute value of <x>."),
            (np.abs, 1),
        ),
        "kmeans": (
            (
                BinaryFunction("kmeans"),
                "(<x>, <k>)",
                "Returns lower bounds of 1 to <k> kmeans clusters.",
            ),
            (kmeans.thresholds, 2),
        ),
        "mask": (
            (
                BinaryFunction("mask"),
                "(<x>, <mask>)",
                "Selects <x> where <mask>, otherwise NaN.",
            ),
            (lambda x, m: np.where(m, x, np.nan), 2),
        ),
        "mean": (
            (UnaryFunction("mean"), "(<x>)", "Returns the mean of <x>."),
            (np.nanmean, 1),
        ),
        "median": (
            (
                UnaryFunction("median"),
                "(<x>)",
                "Returns the median of <x>.",
            ),
            (np.nanmedian, 1),
        ),
        "nantonum": (
            (UnaryFunction("nantonum"), "(<x>)", "Sets nan values to 0."),
            (np.nan_to_num, 1),
        ),
        "normalise": (
            (
                TernaryFunction("normalise"),
                "(<x>, <min>, <max>)",
                "Normalise <x> from from <min> to <max>.",
            ),
            (normalise, 3),
        ),
        "otsu": (
            (
                UnaryFunction("otsu"),
                "(<x>)",
                "Returns Otsu's threshold for <x>.",
            ),
            (otsu, 1),
        ),
        "percentile": (
            (
                BinaryFunction("percentile"),
                "(<x>, <percent>)",
                "Returns the <percent> percentile of <x>.",
            ),
            (np.nanpercentile, 2),
        ),
        "segment": (
            (
                BinaryFunction("segment"),
                "(<x>, <threshold(s)>)",
                "Create a masking image from the given thrshold(s).",
            ),
            (segment_image, 2),
        ),
        "threshold": (
            (
                BinaryFunction("threshold"),
                "(<x>, <value>)",
                "Sets <x> below <value> to NaN.",
            ),
            (lambda x, a: np.where(x > a, x, np.nan), 2),
        ),
    }

    def __init__(self, item: LaserImageItem, view: TabView | None = None):
        super().__init__(item, graphics_label="Preview", view=view)

        self.image: ScaledImageItem | None = None

        self.output = QtWidgets.QLineEdit("Result")
        self.output.setEnabled(False)

        self.lineedit_name = CalculatorName(
            "",
            badnames=[],
            badparser=list(CalculatorTool.functions.keys()),
        )
        self.lineedit_name.revalidate()
        self.lineedit_name.textEdited.connect(self.completeChanged)
        self.lineedit_name.editingFinished.connect(self.refresh)

        self.combo_element = QtWidgets.QComboBox()
        self.combo_element.activated.connect(self.insertVariable)

        functions = [k + v[0][1] for k, v in CalculatorTool.functions.items()]
        tooltips = [v[0][2] for v in CalculatorTool.functions.values()]
        self.combo_function = QtWidgets.QComboBox()
        self.combo_function.addItem("Functions")
        self.combo_function.addItems(functions)
        for i in range(0, len(tooltips)):
            self.combo_function.setItemData(i + 1, tooltips[i], QtCore.Qt.ToolTipRole)
        self.combo_function.activated.connect(self.insertFunction)

        self.reducer = Reducer({})
        self.formula = CalculatorFormula("", variables=[])
        self.formula.textChanged.connect(self.completeChanged)
        self.formula.textChanged.connect(self.refresh)

        self.reducer.operations.update(
            {k: v[1] for k, v in CalculatorTool.functions.items()}
        )
        self.formula.parser.nulls.update(
            {k: v[0][0] for k, v in CalculatorTool.functions.items()}
        )

        layout_combos = QtWidgets.QHBoxLayout()
        layout_combos.addWidget(self.combo_element)
        layout_combos.addWidget(self.combo_function)

        layout_controls = QtWidgets.QFormLayout()
        layout_controls.addRow("Name:", self.lineedit_name)
        layout_controls.addRow("Insert:", layout_combos)
        layout_controls.addRow("Formula:", self.formula)
        layout_controls.addRow("Result:", self.output)
        self.box_controls.setLayout(layout_controls)

        self.initialise()  # refreshes

    def apply(self) -> None:
        name = self.lineedit_name.text()
        data = self.reducer.reduce(self.formula.expr)
        if name in self.item.laser.elements:
            self.item.laser.data[name] = data
        else:
            self.item.laser.add(self.lineedit_name.text(), data)
        # Make sure to repop elements
        self.itemModified.emit(self.item)

        proc = self.item.laser.info.get("Processing", "")
        proc += f"Calculator({name},{self.formula.expr});"
        self.item.laser.info["Processing"] = proc

        self.initialise()

    def initialise(self) -> None:
        elements = self.item.laser.elements
        self.combo_element.clear()
        self.combo_element.addItem("Elements")
        self.combo_element.addItems(elements)

        name = "calc0"
        i = 1
        while name in elements:
            name = f"calc{i}"
            i += 1
        self.lineedit_name.setText(name)
        self.formula.parser.variables = elements
        self.formula.setCompleter(
            QtWidgets.QCompleter(
                list(self.formula.parser.variables)
                + [k + "(" for k in CalculatorTool.functions.keys()]
            )
        )
        self.formula.valid = True
        self.formula.setText(self.item.element())  # refreshes

        # x0, x1, y0, y1 = self.item.laser.config.data_extent(self.item.laser.shape)
        # rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        # self.graphics.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def insertFunction(self, index: int) -> None:
        if index == 0:
            return
        function = self.combo_function.itemText(index)
        function = function[: function.find("(") + 1]
        self.formula.insertPlainText(function)
        self.combo_function.setCurrentIndex(0)
        self.formula.setFocus()

    def insertVariable(self, index: int) -> None:
        if index == 0:
            return
        self.formula.insertPlainText(self.combo_element.itemText(index))
        self.combo_element.setCurrentIndex(0)
        self.formula.setFocus()

    def isComplete(self) -> bool:
        if not self.formula.hasAcceptableInput():
            return False
        if not self.lineedit_name.hasAcceptableInput():
            return False
        return True

    def previewData(self, data: np.ndarray) -> np.ndarray | None:
        self.reducer.variables = {name: data[name] for name in data.dtype.names}
        try:
            result = self.reducer.reduce(self.formula.expr)
            if np.isscalar(result):
                self.output.setText(f"{result:.10g}")
                return None
            elif isinstance(result, np.ndarray) and result.ndim == 1:
                self.output.setText(f"{list(map('{:.4g}'.format, result))}")
                return None
            elif isinstance(result, np.ndarray):
                self.output.setText(f"{result.dtype.name} array: {result.shape}")
                return result
        except (ReducerException, ValueError) as e:
            self.output.setText(str(e))
            return None

    def refresh(self) -> None:
        if not self.isComplete():  # Not ready for update to preview
            return

        data = self.previewData(self.item.laser.get(flat=True, calibrated=False))
        if data is None:
            return
        x0, x1, y0, y1 = self.item.laser.config.data_extent(data.shape)
        rect = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

        vmin, vmax = self.item.options.get_color_range_as_float("<calc>", data)
        data = np.clip(data, vmin, vmax)
        if vmin != vmax:
            data = (data - vmin) / (vmax - vmin)

        table = colortable.get_table(self.item.options.colortable)

        if self.image is not None:
            self.graphics.scene().removeItem(self.image)
        self.image = ScaledImageItem.fromArray(data, rect, table)
        self.image.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False
        )
        self.image.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False
        )
        self.image.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False
        )
        self.graphics.scene().addItem(self.image)

        self.colorbar.updateTable(table, vmin, vmax, "")
        self.graphics.invalidateScene()
