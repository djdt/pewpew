import numpy as np
import logging

from PySide2 import QtCore, QtGui, QtWidgets

from pew.lib import convolve
from pew.lib import filter as fltr
from pew.lib.calc import normalise
from pew.lib.threshold import kmeans_threshold, otsu

from pewpew.actions import qAction, qToolButton
from pewpew.lib.pratt import Parser, ParserException, Reducer, ReducerException
from pewpew.lib.pratt import BinaryFunction, UnaryFunction, TernaryFunction

from pewpew.widgets.canvases import BasicCanvas, LaserImageCanvas
from pewpew.widgets.ext import ValidColorLineEdit, ValidColorTextEdit
from pewpew.widgets.laser import LaserWidget
from pewpew.widgets.tools import ToolWidget

from pewpew.validators import DecimalValidator, LimitValidator, OddIntValidator

from typing import List, Tuple


# TODO
# Add some kind of indicator for if all data or just current isotope changed

logger = logging.getLogger(__name__)


class EditTool(ToolWidget):
    METHODS = ["Calculator", "Convolve", "Deconvolve", "Filter", "Transform"]

    def __init__(self, widget: LaserWidget):
        super().__init__(widget)

        self.setWindowTitle("Calculator Tool")

        self.rotate = 0
        self.flip_horizontal = False
        self.flip_vertical = False

        self.action_transform_flip_horizontal = qAction(
            "object-flip-horizontal",
            "Flip Horizontal",
            "Flip the image about vertical axis.",
            self.actionTransformFlipHorz,
        )
        self.action_transform_flip_vertical = qAction(
            "object-flip-vertical",
            "Flip Vertical",
            "Flip the image about horizontal axis.",
            self.actionTransformFlipVert,
        )
        self.action_transform_rotate_left = qAction(
            "object-rotate-left",
            "Rotate Left",
            "Rotate the image 90° counter clockwise.",
            self.actionTransformRotateLeft,
        )
        self.action_transform_rotate_right = qAction(
            "object-rotate-right",
            "Rotate Right",
            "Rotate the image 90° clockwise.",
            self.actionTransformRotateRight,
        )

        self.canvas = LaserImageCanvas(
            self.viewspace.options, move_button=1, parent=self
        )
        self.canvas.drawFigure()
        self.canvas.cursorClear.connect(self.widget.clearCursorStatus)
        self.canvas.cursorMoved.connect(self.widget.updateCursorStatus)
        self.canvas.view_limits = self.widget.canvas.view_limits

        self.button_transform_flip_horizontal = qToolButton(
            action=self.action_transform_flip_horizontal
        )
        self.button_transform_flip_vertical = qToolButton(
            action=self.action_transform_flip_vertical
        )
        self.button_transform_rotate_left = qToolButton(
            action=self.action_transform_rotate_left
        )
        self.button_transform_rotate_right = qToolButton(
            action=self.action_transform_rotate_right
        )

        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(EditTool.METHODS)

        self.calculator_method = CalculatorMethod(self)
        self.calculator_method.inputChanged.connect(self.completeChanged)
        self.calculator_method.inputChanged.connect(self.refresh)

        self.convolve_method = ConvolveMethod(self)
        self.convolve_method.inputChanged.connect(self.completeChanged)
        self.convolve_method.inputChanged.connect(self.refresh)

        self.deconvolve_method = DeconvolveMethod(self)
        self.deconvolve_method.inputChanged.connect(self.completeChanged)
        self.deconvolve_method.inputChanged.connect(self.refresh)

        self.filter_method = FilterMethod(self)
        self.filter_method.inputChanged.connect(self.completeChanged)
        self.filter_method.inputChanged.connect(self.refresh)

        self.transform_method = TransformMethod(self)
        self.transform_method.inputChanged.connect(self.completeChanged)
        self.transform_method.inputChanged.connect(self.refresh)

        self.method_stack = QtWidgets.QStackedWidget()
        self.method_stack.addWidget(self.calculator_method)
        self.method_stack.addWidget(self.convolve_method)
        self.method_stack.addWidget(self.deconvolve_method)
        self.method_stack.addWidget(self.filter_method)
        self.method_stack.addWidget(self.transform_method)
        # Make sure to add the stack widgets in right order!
        self.combo_method.currentIndexChanged.connect(self.setCurrentMethod)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.currentIndexChanged.connect(self.completeChanged)
        self.combo_isotope.currentIndexChanged.connect(self.refresh)
        self.combo_isotope.setEnabled(False)

        layout_methods = QtWidgets.QVBoxLayout()
        layout_methods.addWidget(self.combo_method, 0, QtCore.Qt.AlignLeft)
        layout_methods.addWidget(self.method_stack)

        canvas_box = QtWidgets.QGroupBox("Preview")
        layout_canvas_box = QtWidgets.QVBoxLayout()
        layout_canvas_box.addWidget(self.canvas)
        canvas_box.setLayout(layout_canvas_box)

        layout_canvas = QtWidgets.QVBoxLayout()
        layout_canvas.addWidget(canvas_box)

        layout_canvas_bar = QtWidgets.QHBoxLayout()
        layout_canvas_bar.addWidget(self.button_transform_rotate_left)
        layout_canvas_bar.addWidget(self.button_transform_rotate_right)
        layout_canvas_bar.addWidget(self.button_transform_flip_horizontal)
        layout_canvas_bar.addWidget(self.button_transform_flip_vertical)
        layout_canvas_bar.addWidget(self.combo_isotope, 0, QtCore.Qt.AlignRight)

        layout_canvas.addLayout(layout_canvas_bar)

        self.layout_main.setDirection(QtWidgets.QBoxLayout.LeftToRight)
        self.layout_main.addLayout(layout_methods, 0)
        self.layout_main.addLayout(layout_canvas, 1)

        self.widgetChanged()

    def actionTransformFlipHorz(self) -> None:
        self.flip_horizontal = not self.flip_horizontal
        self.refresh()

    def actionTransformFlipVert(self) -> None:
        self.flip_vertical = not self.flip_vertical
        self.refresh()

    def actionTransformRotateLeft(self) -> None:
        self.rotate = (self.rotate + 3) % 4
        self.refresh()

    def actionTransformRotateRight(self) -> None:
        self.rotate = (self.rotate + 1) % 4
        self.refresh()

    def apply(self) -> None:
        i = self.method_stack.currentIndex()
        logger.info(f"Applying {EditTool.METHODS[i]} to {self.widget.laser.name}.")
        stack = self.method_stack.widget(i)
        if stack.isComplete():
            stack.apply()
        self.widgetChanged()

    def isComplete(self) -> bool:
        return self.method_stack.currentWidget().isComplete()

    def previewData(self, isotope: str = None) -> np.ndarray:
        data = self.widget.laser.get(isotope, flat=True, calibrated=False)

        if self.flip_horizontal:
            data = np.flip(data, axis=1)
        if self.flip_vertical:
            data = np.flip(data, axis=0)
        if self.rotate != 0:
            data = np.rot90(data, k=self.rotate, axes=(1, 0))
        return data

    def refresh(self) -> None:
        stack: MethodStackWidget = self.method_stack.currentWidget()
        if not stack.isComplete():  # Not ready for update to preview
            return

        isotope = self.combo_isotope.currentText()
        if stack.full_data:
            data = stack.previewData(self.previewData())
        else:
            data = stack.previewData(self.previewData(isotope))
        if data is None:
            return

        extent = self.widget.laser.config.data_extent(data.shape)
        # Only change the view if new or the laser extent has changed (i.e. conf edit)
        if self.canvas.extent != extent:
            self.canvas.view_limits = self.canvas.extentForAspect(extent)

        self.canvas.drawData(
            data, extent, isotope=isotope,
        )
        if self.canvas.viewoptions.canvas.colorbar:
            self.canvas.drawColorbar(self.widget.laser.calibration[isotope].unit)

        if self.canvas.viewoptions.canvas.label:
            self.canvas.drawLabel(isotope)
        elif self.canvas.label is not None:
            self.canvas.label.remove()
            self.canvas.label = None

        if self.canvas.viewoptions.canvas.scalebar:
            self.canvas.drawScalebar()
        elif self.canvas.scalebar is not None:
            self.canvas.scalebar.remove()
            self.canvas.scalebar = None

        self.canvas.draw_idle()

    def setCurrentMethod(self, method: int) -> None:
        self.method_stack.setCurrentIndex(method)
        self.combo_isotope.setEnabled(not self.method_stack.currentWidget().full_data)
        self.refresh()

    def widgetChanged(self) -> None:
        # Prevent currentIndexChanged being emitted
        self.combo_isotope.blockSignals(True)
        self.combo_isotope.clear()
        self.combo_isotope.addItems(self.widget.laser.isotopes)
        self.combo_isotope.setCurrentText(self.widget.combo_isotope.currentText())
        self.combo_isotope.blockSignals(False)

        for i in range(self.method_stack.count()):
            self.method_stack.widget(i).initialise()

        self.refresh()


class MethodStackWidget(QtWidgets.QGroupBox):
    inputChanged = QtCore.Signal()

    def __init__(self, parent: EditTool, full_data: bool = False):
        super().__init__("Options", parent)
        self.edit = parent
        self.full_data = full_data

    def apply(self) -> None:
        if self.edit.widget.is_srr:
            logger.warn("Method not implemented for SRR data.")
            QtWidgets.QMessageBox.warning("Not yet implemented for SRR data.")
            return
        isotope = self.edit.combo_isotope.currentText()
        data = self.edit.widget.laser.data[isotope]
        self.edit.widget.laser.data[isotope] = self.previewData(data)

    def initialise(self) -> None:
        pass

    def isComplete(self) -> bool:
        return True

    def previewData(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CalculatorName(ValidColorLineEdit):
    def __init__(
        self,
        text: str,
        badnames: List[str],
        badparser: List[str],
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(text, parent)

        self.badchars = " +-=*/\\^<>!()[]"
        self.badnames = badnames
        self._badnames = ["nan", "if", "then", "else"]
        self._badnames.extend(badparser)

    def hasAcceptableInput(self) -> bool:
        if self.text() == "":
            return False
        if any(c in self.text() for c in self.badchars):
            return False
        if self.text() in self._badnames:
            return False
        if self.text() in self.badnames:
            return False
        return True


class CalculatorFormula(ValidColorTextEdit):
    def __init__(
        self, text: str, variables: List[str], parent: QtWidgets.QWidget = None
    ):
        super().__init__(text, parent)
        self.textChanged.disconnect(self.revalidate)
        self.textChanged.connect(self.calculate)
        self.parser = Parser(variables)
        self.expr = ""

        self.cgood = self.palette().color(QtGui.QPalette.Base)
        self.cbad = QtGui.QColor.fromRgb(255, 172, 172)

    def hasAcceptableInput(self) -> bool:
        return self.expr != ""

    def calculate(self) -> None:
        try:
            self.expr = self.parser.parse(self.toPlainText())
        except ParserException:
            self.expr = ""
        self.revalidate()


class CalculatorMethod(MethodStackWidget):
    parser_functions = {
        "abs": (UnaryFunction("abs"), "(<x>)", "The absolute value of <x>."),
        "kmeans": (
            BinaryFunction("kmeans"),
            "(<x>, <k>)",
            "Returns lower bounds of 1 to <k> kmeans clusters.",
        ),
        "mean": (UnaryFunction("mean"), "(<x>)", "Returns the mean of <x>."),
        "median": (UnaryFunction("median"), "(<x>)", "Returns the median of <x>.",),
        "normalise": (
            TernaryFunction("normalise"),
            "(<x>, <min>, <max>)",
            "Normalise <x> from from <min> to <max>.",
        ),
        "otsu": (UnaryFunction("otsu"), "(<x>)", "Returns Otsu's threshold for <x>.",),
        # "multiotsu": (BinaryFunction("multiotsu"), "(<x>, <t>)", "Returns <t> thresholds for <x>.",),
        "percentile": (
            BinaryFunction("percentile"),
            "(<x>, <percent>)",
            "Returns the <percent> percentile of <x>.",
        ),
        "threshold": (
            BinaryFunction("threshold"),
            "(<x>, <value>)",
            "Sets <x> below <value> to NaN.",
        ),
    }
    reducer_functions = {
        "abs": (np.abs, 1),
        "kmeans": (kmeans_threshold, 2),
        "mean": (np.nanmean, 1),
        "median": (np.nanmedian, 1),
        "normalise": (normalise, 3),
        "otsu": (otsu, 1),
        # "multiotsu": (multiotsu, 2),
        "percentile": (np.nanpercentile, 2),
        "threshold": (lambda x, a: np.where(x > a, x, np.nan), 2),
    }

    def __init__(self, parent: EditTool):
        super().__init__(parent, full_data=True)
        self.output = QtWidgets.QLineEdit("Result")
        self.output.setEnabled(False)

        self.lineedit_name = CalculatorName(
            "", badnames=[], badparser=list(CalculatorMethod.parser_functions.keys()),
        )
        self.lineedit_name.revalidate()
        self.lineedit_name.textEdited.connect(self.inputChanged)

        self.combo_isotope = QtWidgets.QComboBox()
        self.combo_isotope.activated.connect(self.insertVariable)

        functions = [k + v[1] for k, v in CalculatorMethod.parser_functions.items()]
        tooltips = [v[2] for v in CalculatorMethod.parser_functions.values()]
        self.combo_function = QtWidgets.QComboBox()
        self.combo_function.addItem("Functions")
        self.combo_function.addItems(functions)
        for i in range(0, len(tooltips)):
            self.combo_function.setItemData(i + 1, tooltips[i], QtCore.Qt.ToolTipRole)
        self.combo_function.activated.connect(self.insertFunction)

        self.reducer = Reducer({})
        self.formula = CalculatorFormula("", variables=[])
        self.formula.textChanged.connect(self.inputChanged)

        self.reducer.operations.update(CalculatorMethod.reducer_functions)
        self.formula.parser.nulls.update(
            {k: v[0] for k, v in CalculatorMethod.parser_functions.items()}
        )

        layout_combos = QtWidgets.QHBoxLayout()
        layout_combos.addWidget(self.combo_isotope)
        layout_combos.addWidget(self.combo_function)

        layout_grid = QtWidgets.QGridLayout()
        layout_grid.addWidget(QtWidgets.QLabel("Name:"), 0, 0)
        layout_grid.addWidget(self.lineedit_name, 0, 1)
        layout_grid.addWidget(QtWidgets.QLabel("Insert:"), 1, 0)
        layout_grid.addLayout(layout_combos, 1, 1)
        layout_grid.addWidget(QtWidgets.QLabel("Formula:"), 2, 0)
        layout_grid.addWidget(self.formula, 2, 1, 1, 1)
        layout_grid.addWidget(QtWidgets.QLabel("Result:"), 3, 0)
        layout_grid.addWidget(self.output, 3, 1)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_grid)
        layout_main.addStretch(0)
        self.setLayout(layout_main)

    def apply(self) -> None:
        name = self.lineedit_name.text()
        data = self.reducer.reduce(self.formula.expr)
        if name in self.edit.widget.laser.isotopes:
            self.edit.widget.laser.data[name] = data
        else:
            self.edit.widget.laser.add(self.lineedit_name.text(), data)
        # Make sure to repop isotopes
        self.edit.widget.populateIsotopes()

    def initialise(self) -> None:
        isotopes = self.edit.widget.laser.isotopes
        self.combo_isotope.clear()
        self.combo_isotope.addItem("Isotopes")
        self.combo_isotope.addItems(isotopes)

        name = "calc0"
        i = 1
        while name in isotopes:
            name = f"calc{i}"
            i += 1
        self.lineedit_name.setText(name)
        self.formula.parser.variables = isotopes
        self.formula.valid = True
        self.formula.setText(self.edit.combo_isotope.currentText())

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
        self.formula.insertPlainText(self.combo_isotope.itemText(index))
        self.combo_isotope.setCurrentIndex(0)
        self.formula.setFocus()

    def isComplete(self) -> bool:
        if not self.formula.hasAcceptableInput():
            return False
        if not self.lineedit_name.hasAcceptableInput():
            return False
        return True

    def previewData(self, data: np.ndarray) -> np.ndarray:
        self.reducer.variables = {name: data[name] for name in data.dtype.names}
        try:
            data = self.reducer.reduce(self.formula.expr)
            if np.isscalar(data):
                self.output.setText(f"{data:.10g}")
                return None
            elif isinstance(data, np.ndarray) and data.ndim == 1:
                self.output.setText(f"{list(map('{:.4g}'.format, data))}")
                return None
            elif isinstance(data, np.ndarray):
                self.output.setText("")
                return data
        except (ReducerException, ValueError) as e:
            self.output.setText(str(e))
            return None


class ConvolveKernelCanvas(BasicCanvas):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__((1.0, 1.0), parent=parent)

    def drawFigure(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot()
        # self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

    def drawKernel(self, kernel: np.ndarray):
        self.ax.clear()
        self.ax.plot(kernel[:, 0], kernel[:, 1], color="red")
        self.draw_idle()

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(150, 150)


class ConvolveMethod(MethodStackWidget):
    kernels: dict = {
        "Beta": {
            "psf": convolve.beta,
            "params": [("α", 2.0, (0.0, np.inf)), ("β", 5.0, (0.0, np.inf))],
        },
        # "Cauchy": {
        #     "psf": convolve.cauchy,
        #     "params": [("γ", 1.0, (0.0, np.inf)), ("x₀", 0.0, (-np.inf, np.inf))],
        # },
        "Exponential": {
            "psf": convolve.exponential,
            "params": [("λ", 1.0, (0.0, np.inf))],
        },
        "Inverse-gamma": {
            "psf": convolve.inversegamma,
            "params": [("α", 1.0, (0.0, np.inf)), ("β", 1.0, (0.0, np.inf))],
        },
        "Laplace": {
            "psf": convolve.laplace,
            "params": [("b", 0.5, (0.0, np.inf)), ("μ", 0.0, (-np.inf, np.inf))],
        },
        # "Log-Cauchy": {
        #     "psf": convolve.logcauchy,
        #     "params": [("σ", 1.0, (0, np.inf)), ("μ", 0.0, (-np.inf, np.inf))],
        # },
        "Log-Laplace": {
            "psf": convolve.loglaplace,
            "params": [("b", 0.5, (0.0, np.inf)), ("μ", 0.0, (-np.inf, np.inf))],
        },
        "Log-normal": {
            "psf": convolve.lognormal,
            "params": [("σ", 1.0, (0.0, np.inf)), ("μ", 0.0, (-np.inf, np.inf))],
        },
        "Gaussian": {
            "psf": convolve.normal,
            "params": [("σ", 1.0, (0.0, np.inf)), ("μ", 0.0, (-np.inf, np.inf))],
        },
        "Super-Gaussian": {
            "psf": convolve.super_gaussian,
            "params": [
                ("σ", 1.0, (0.0, np.inf)),
                ("μ", 0.0, (-np.inf, np.inf)),
                ("P", 2.0, (-np.inf, np.inf)),
            ],
        },
        "Triangular": {
            "psf": convolve.triangular,
            "params": [("a", -2.0, (-np.inf, 0.0)), ("b", 2.0, (0.0, np.inf))],
        },
    }

    def __init__(self, parent: EditTool):
        super().__init__(parent)

        self.combo_horizontal = QtWidgets.QComboBox()
        self.combo_horizontal.addItems(["No", "Left to Right", "Right to Left"])
        self.combo_horizontal.activated.connect(self.inputChanged)
        self.combo_vertical = QtWidgets.QComboBox()
        self.combo_vertical.addItems(["No", "Top to Bottom", "Bottom to Top"])
        self.combo_vertical.activated.connect(self.inputChanged)

        # kernel type --- gaussian, etc
        self.combo_kernel = QtWidgets.QComboBox()
        self.combo_kernel.addItems(ConvolveMethod.kernels.keys())
        self.combo_kernel.setCurrentText("Gaussian")
        self.combo_kernel.activated.connect(self.kernelChanged)
        self.combo_kernel.activated.connect(self.inputChanged)
        # kernel size
        self.lineedit_ksize = ValidColorLineEdit()
        self.lineedit_ksize.setValidator(QtGui.QIntValidator(2, 999))
        self.lineedit_ksize.textEdited.connect(self.inputChanged)
        self.lineedit_kscale = ValidColorLineEdit()
        self.lineedit_kscale.setValidator(DecimalValidator(1e-9, 1e9, 4))
        self.lineedit_kscale.textEdited.connect(self.inputChanged)
        # kernel param
        nparams = np.amax([len(k["params"]) for k in ConvolveMethod.kernels.values()])
        self.label_kparams = [QtWidgets.QLabel() for i in range(nparams)]
        self.lineedit_kparams = [ValidColorLineEdit() for i in range(nparams)]
        for le in self.lineedit_kparams:
            le.textEdited.connect(self.inputChanged)
            le.setValidator(LimitValidator(0.0, 0.0, 4))

        # kernel preview
        self.canvas_kernel = ConvolveKernelCanvas(self)
        self.canvas_kernel.drawFigure()

        layout_combo = QtWidgets.QFormLayout()
        layout_combo.addRow("Horizontal:", self.combo_horizontal)
        layout_combo.addRow("Vertical:", self.combo_vertical)

        layout_kernel = QtWidgets.QFormLayout()
        layout_kernel.addWidget(self.combo_kernel)
        layout_kernel.addRow("Size:", self.lineedit_ksize)
        layout_kernel.addRow("Scale:", self.lineedit_kscale)
        for i in range(len(self.label_kparams)):
            layout_kernel.addRow(self.label_kparams[i], self.lineedit_kparams[i])

        box_kernel = QtWidgets.QGroupBox()
        box_kernel.setLayout(layout_kernel)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_combo)
        layout_main.addWidget(box_kernel)
        layout_main.addWidget(self.canvas_kernel)
        self.setLayout(layout_main)

    @property
    def kscale(self) -> float:
        return 1.0 / float(self.lineedit_kscale.text())

    @property
    def ksize(self) -> int:
        return int(self.lineedit_ksize.text())

    @property
    def kparams(self) -> List[float]:
        return [float(le.text()) for le in self.lineedit_kparams if le.isEnabled()]

    def initialise(self) -> None:
        if not self.lineedit_ksize.hasAcceptableInput():
            self.lineedit_ksize.setText("8")
        if not self.lineedit_kscale.hasAcceptableInput():
            self.lineedit_kscale.setText("1.0")

        self.kernelChanged()

    def kernelChanged(self) -> None:
        kernel = ConvolveMethod.kernels[self.combo_kernel.currentText()]
        # Clear all the current params
        for le in self.label_kparams:
            le.setVisible(False)
            le.setEnabled(False)
        for le in self.lineedit_kparams:
            le.setVisible(False)
            le.setEnabled(False)

        params: List[Tuple[str, float, Tuple]] = kernel["params"]

        for i, (symbol, default, range) in enumerate(params):
            self.label_kparams[i].setText(f"{symbol}:")
            self.label_kparams[i].setVisible(True)
            self.lineedit_kparams[i].validator().setRange(range[0], range[1], 4)
            self.lineedit_kparams[i].setVisible(True)
            self.lineedit_kparams[i].setEnabled(True)
            # Keep input that's still valid
            if not self.lineedit_kparams[i].hasAcceptableInput():
                self.lineedit_kparams[i].setText(str(default))
                self.lineedit_kparams[i].revalidate()

    def isComplete(self) -> bool:
        if not self.lineedit_ksize.hasAcceptableInput():
            return False
        if not self.lineedit_kscale.hasAcceptableInput():
            return False
        if not all(
            le.hasAcceptableInput() for le in self.lineedit_kparams if le.isVisible()
        ):
            return False
        return True

    def previewData(self, data: np.ndarray) -> np.ndarray:
        # Preview the kernel too
        psf = ConvolveMethod.kernels[self.combo_kernel.currentText()]["psf"]
        kernel = psf(self.ksize, *self.kparams, scale=self.kscale)

        if kernel.sum() == 0:
            self.canvas_kernel.ax.clear()
            self.canvas_kernel.draw_idle()
            return None  # Invalid kernel
        else:
            self.canvas_kernel.drawKernel(kernel)

        hmode = self.combo_horizontal.currentText()
        hslice = slice(None, None, -1 if hmode == "Right to Left" else 1)
        vmode = self.combo_vertical.currentText()
        vslice = slice(None, None, -1 if hmode == "Bottom to Top" else 1)

        data = data[vslice, hslice]

        kernel = kernel[:, 1]

        if hmode != "No":
            data = np.apply_along_axis(convolve.convolve, 1, data, kernel, mode="pad")
        if vmode != "No":
            data = np.apply_along_axis(convolve.convolve, 0, data, kernel, mode="pad")

        return data[vslice, hslice]


class DeconvolveMethod(ConvolveMethod):
    def previewData(self, data: np.ndarray) -> np.ndarray:
        # Preview the kernel too
        psf = ConvolveMethod.kernels[self.combo_kernel.currentText()]["psf"]
        kernel = psf(self.ksize, *self.kparams, scale=self.kscale)

        if kernel.sum() == 0:
            self.canvas_kernel.ax.clear()
            self.canvas_kernel.draw_idle()
            return None  # Invalid kernel
        else:
            self.canvas_kernel.drawKernel(kernel)

        hmode = self.combo_horizontal.currentText()
        hslice = slice(None, None, -1 if hmode == "Right to Left" else 1)
        vmode = self.combo_vertical.currentText()
        vslice = slice(None, None, -1 if hmode == "Bottom to Top" else 1)

        data = data[vslice, hslice]

        kernel = kernel[:, 1]

        if hmode != "No":
            data = np.apply_along_axis(
                convolve.deconvolve, 1, data, kernel, mode="same"
            )
        if vmode != "No":
            data = np.apply_along_axis(
                convolve.deconvolve, 0, data, kernel, mode="same"
            )

        return data[vslice, hslice]


class FilterMethod(MethodStackWidget):
    filters: dict = {
        # "Low-pass": {
        #     "filter": fltrs.low_pass_filter,
        #     "params": [("d", 0.5, (0.0, 1.0))],
        #     "desc": ["Filter if low pass changes value d amount."],
        # },
        "Mean": {
            "filter": fltr.mean_filter,
            "params": [("σ", 3.0, (0.0, np.inf))],
            "desc": ["Filter if > σ stddevs from mean."],
        },
        "Median": {
            "filter": fltr.median_filter,
            "params": [("M", 3.0, (0.0, np.inf))],
            "desc": ["Filter if > M medians from median."],
        },
    }

    def __init__(self, parent: EditTool):
        super().__init__(parent)

        self.combo_filter = QtWidgets.QComboBox()
        self.combo_filter.addItems(FilterMethod.filters.keys())
        self.combo_filter.setCurrentText("Median")
        self.combo_filter.activated.connect(self.filterChanged)
        self.combo_filter.activated.connect(self.inputChanged)

        self.lineedit_fsize = ValidColorLineEdit()
        self.lineedit_fsize.setValidator(OddIntValidator(3, 21))
        self.lineedit_fsize.textEdited.connect(self.inputChanged)

        nparams = np.amax([len(f["params"]) for f in FilterMethod.filters.values()])
        self.label_fparams = [QtWidgets.QLabel() for i in range(nparams)]
        self.lineedit_fparams = [ValidColorLineEdit() for i in range(nparams)]
        for le in self.lineedit_fparams:
            le.textEdited.connect(self.inputChanged)
            le.setValidator(LimitValidator(0.0, 0.0, 4))

        layout_filter = QtWidgets.QFormLayout()
        layout_filter.addWidget(self.combo_filter)
        layout_filter.addRow("Size:", self.lineedit_fsize)
        for i in range(len(self.label_fparams)):
            layout_filter.addRow(self.label_fparams[i], self.lineedit_fparams[i])

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addLayout(layout_filter)

        self.setLayout(layout_main)

    @property
    def fsize(self) -> int:
        return int(self.lineedit_fsize.text())

    @property
    def fparams(self) -> List[float]:
        return [float(le.text()) for le in self.lineedit_fparams if le.isVisible()]

    def initialise(self) -> None:
        if not self.lineedit_fsize.hasAcceptableInput():
            self.lineedit_fsize.setText("5")

        self.filterChanged()

    def filterChanged(self) -> None:
        filter_ = FilterMethod.filters[self.combo_filter.currentText()]
        # Clear all the current params
        for le in self.label_fparams:
            le.setVisible(False)
        for le in self.lineedit_fparams:
            le.setVisible(False)

        params: List[Tuple[str, float, Tuple]] = filter_["params"]

        for i, (symbol, default, range) in enumerate(params):
            self.label_fparams[i].setText(f"{symbol}:")
            self.label_fparams[i].setVisible(True)
            self.lineedit_fparams[i].validator().setRange(range[0], range[1], 4)
            self.lineedit_fparams[i].setVisible(True)
            self.lineedit_fparams[i].setToolTip(filter_["desc"][i])
            # keep input that's still valid
            if not self.lineedit_fparams[i].hasAcceptableInput():
                self.lineedit_fparams[i].setText(str(default))
                self.lineedit_fparams[i].revalidate()

    def isComplete(self) -> bool:
        if not self.lineedit_fsize.hasAcceptableInput():
            return False
        if not all(
            le.hasAcceptableInput() for le in self.lineedit_fparams if le.isVisible()
        ):
            return False
        return True

    def previewData(self, data: np.ndarray) -> np.ndarray:
        filter_ = FilterMethod.filters[self.combo_filter.currentText()]["filter"]
        return filter_(data, (self.fsize, self.fsize), *self.fparams)


class TransformMethod(MethodStackWidget):
    def __init__(self, parent: EditTool):
        super().__init__(parent)

        self.lineedit_scale = ValidColorLineEdit("1")
        self.lineedit_scale.setValidator(QtGui.QIntValidator(1, 10))
        self.lineedit_scale.textEdited.connect(self.inputChanged)

        self.lineedit_trims = [ValidColorLineEdit("0") for i in range(4)]
        for le in self.lineedit_trims:
            le.setValidator(QtGui.QIntValidator(0, 9999))
            le.textEdited.connect(self.inputChanged)

        layout_trim_x = QtWidgets.QHBoxLayout()
        layout_trim_x.addWidget(self.lineedit_trims[0])
        layout_trim_x.addWidget(QtWidgets.QLabel("x"))
        layout_trim_x.addWidget(self.lineedit_trims[1])
        layout_trim_y = QtWidgets.QHBoxLayout()
        layout_trim_y.addWidget(self.lineedit_trims[2])
        layout_trim_y.addWidget(QtWidgets.QLabel("x"))
        layout_trim_y.addWidget(self.lineedit_trims[3])

        layout = QtWidgets.QFormLayout()
        layout.addRow("Scale", self.lineedit_scale)
        layout.addRow("Trim X", layout_trim_x)
        layout.addRow("Trim Y", layout_trim_y)

        self.setLayout(layout)

    @property
    def scale(self) -> int:
        return int(self.lineedit_scale.text())

    @property
    def trim(self) -> Tuple[int, int, int, int]:
        return tuple(int(le.text()) for le in self.lineedit_trims)  # type: ignore

    def apply(self) -> None:
        if self.edit.widget.is_srr:
            logger.warn("Method not implemented for SRR data.")
            QtWidgets.QMessageBox.warning("Not yet implemented for SRR data.")
            return
        data = self.edit.widget.laser.data
        trim = self.trim
        shape = (
            data.shape[0] * self.scale - (trim[2] + trim[3]),
            data.shape[1] * self.scale - (trim[0] + trim[1]),
        )
        new_data = np.empty(shape, dtype=data.dtype)
        for name in new_data.dtype.names:
            new_data[name] = self.previewData(data[name])
        self.edit.widget.laser.data = new_data

    def initialise(self) -> None:
        self.lineedit_scale.setText("1")
        for le in self.lineedit_trims:
            le.setText("0")

    def isComplete(self) -> bool:
        if not self.lineedit_scale.hasAcceptableInput():
            return False
        if not all(le.hasAcceptableInput() for le in self.lineedit_trims):
            return False
        trim = self.trim
        if trim[0] + trim[1] > self.edit.widget.laser.shape[1] * self.scale - 1:
            self.lineedit_trims[0].setValid(False)
            self.lineedit_trims[1].setValid(False)
            return False
        else:
            self.lineedit_trims[0].setValid(True)
            self.lineedit_trims[1].setValid(True)

        if trim[2] + trim[3] > self.edit.widget.laser.shape[0] * self.scale - 1:
            self.lineedit_trims[2].setValid(False)
            self.lineedit_trims[3].setValid(False)
            return False
        else:
            self.lineedit_trims[2].setValid(True)
            self.lineedit_trims[3].setValid(True)
        return True

    def previewData(self, data: np.ndarray) -> np.ndarray:
        data = np.repeat(np.repeat(data, self.scale, axis=0), self.scale, axis=1)
        trim = self.trim
        return data[
            trim[2] : data.shape[0] - trim[3], trim[0] : data.shape[1] - trim[1],
        ]
