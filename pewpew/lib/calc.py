import numpy as np
import re

from typing import List, Union, Tuple


def rolling_mean_filter(
    x: np.ndarray, window: Tuple[int, int], threshold: int = 3
) -> np.ndarray:
    """Rolling mean filter an array.

    The window size should be an integer divisor of the array size.

    Args:
        window: Shape of the rolling window.
        threshold: σ's value must be from mean to be an outlier.

    """
    x = x.copy()
    # Create view
    roll = rolling_window(x, window)
    # Distance from mean (in stdevs)
    means = np.mean(roll, axis=(2, 3), keepdims=True)
    stds = np.std(roll, axis=(2, 3), keepdims=True)
    diffs = np.abs(roll - means) / stds
    # Recalculate mean, without outliers
    roll[diffs > threshold] = np.nan
    means = np.nanmean(roll, axis=(2, 3), keepdims=True)
    # Replace all outliers and copy back into view
    np.copyto(roll, means, where=diffs > threshold)
    return x


def rolling_median_filter(
    x: np.ndarray, window: Tuple[int, int], threshold: int = 3
) -> np.ndarray:
    """Rolling median filter an array.

    The window size should be an integer divisor of the array size.

    Args:
        window: Shape of the rolling window.
        threshold: N-distance's from median to be considered outlier.

    """
    x = x.copy()
    # Create view
    roll = rolling_window(x, window)
    # Distance from the median
    medians = np.median(roll, axis=(2, 3), keepdims=True)
    diffs = np.abs(roll - medians)
    # Median difference
    median_diffs = np.median(diffs, axis=(2, 3), keepdims=True)
    # Normalise differences
    diffs = np.divide(diffs, median_diffs, where=median_diffs != 0)
    # Replace all over threshold and copy back into view
    np.copyto(roll, medians, where=diffs > threshold)

    return x


def rolling_window(x: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    """Create non-overlapping views into a array.

    Args:
        x: The array.
        window: The size of the view.

    Returns:
        An array of views.
    """
    x = np.ascontiguousarray(x)
    shape = tuple(np.array(x.shape) // window) + window
    strides = tuple(np.array(x.strides) * window) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def rolling_window_step(
    x: np.ndarray, window: Tuple[int, int], step: int
) -> np.ndarray:
    """Create overlapping views into a array.

    Args:
        x: The array.
        window: The size of the view.
        step: Offset of the next window.

    Returns:
        An array of views.
    """
    x = np.ascontiguousarray(x)
    slices = tuple(slice(None, None, st) for st in (step,) * x.ndim)
    shape = tuple(
        list(((np.array(x.shape) - np.array(window)) // np.array(step)) + 1)
        + list(window)
    )
    strides = tuple(list(x[slices].strides) + list(x.strides))
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


class FormulaException(Exception):
    pass


class FormulaParser(object):
    def __init__(self, variables: dict):
        number_token = "\\d+\\.?\\d*"
        variable_token = "\\d*[a-zA-Z]+\\d*"

        self.regexp_isvar = re.compile(variable_token)
        self.regexp_isnum = re.compile(number_token)
        self.regexp_tokenise = re.compile(
            f"\\s*({variable_token}|{number_token}|\\S)\\s*"
        )

        self.operations = {
            "+": np.add,
            "-": np.subtract,
            "*": np.multiply,
            "/": np.divide,
            "^": np.power,
        }

        self.variables = variables

    def isVariable(self, string: str) -> bool:
        return self.regexp_isvar.match(string) is not None

    def isNumber(self, string: str) -> bool:
        return self.regexp_isnum.match(string) is not None

    def findOuterParenthesis(self, tokens: List) -> np.ndarray:
        try:
            start = tokens.index("(")
            end = next(i for i in reversed(range(len(tokens))) if tokens[i] == ")")
        except (ValueError, StopIteration):
            raise FormulaException("Mismatched parenthesis.")
        return start, end

    def applyOp(
        self, a: Union[float, np.ndarray], op: str, b: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        callable = FormulaParser.OPERATIONS[op]
        return callable(a, b)

    def parseType(self, tokens: List[str]) -> dict:
        if len(tokens) == 0:
            raise FormulaException(f"Unexpected end to input.")
        t = tokens.pop(0)
        if self.isNumber(t):
            return dict(type="num", value=t)
        elif self.isVariable(t):
            return dict(type="var", value=t)
        elif t == "(":
            expr = self.parseAddition(tokens)
            if len(tokens) == 0 or tokens[0] != ")":
                raise FormulaException("Mismatched parenthesis.")
            tokens.pop(0)
            return expr
        else:
            raise FormulaException(f"Unexpected input '{t}'.")

    def parsePowers(self, tokens: List[str]) -> dict:
        expr = self.parseType(tokens)
        while len(tokens) > 0 and tokens[0] == "^":
            t = tokens.pop(0)
            rhs = self.parseType(tokens)
            expr = dict(type=t, left=expr, right=rhs)
        return expr

    def parseMultiplication(self, tokens: List[str]) -> dict:
        expr = self.parsePowers(tokens)
        while len(tokens) > 0 and tokens[0] in "*/":
            t = tokens.pop(0)
            rhs = self.parsePowers(tokens)
            expr = dict(type=t, left=expr, right=rhs)
        return expr

    def parseAddition(self, tokens: List[str]) -> dict:
        expr = self.parseMultiplication(tokens)
        while len(tokens) > 0 and tokens[0] in "+-":
            t = tokens.pop(0)
            rhs = self.parseMultiplication(tokens)
            expr = dict(type=t, left=expr, right=rhs)
        return expr

    def parse(self, string: str) -> dict:
        tokens = self.regexp_tokenise.findall(string)

        result = self.parseAddition(tokens)

        if len(tokens) != 0:
            raise FormulaException(f"Unexpected input '{tokens[0]}'.")

        return result

    def validate(self, expr: dict) -> bool:
        if expr["type"] in "+-*/^":
            return self.validate(expr["left"]) and self.validate(expr["right"])
        elif expr["type"] == "num":
            return True
        elif expr["type"] == "var":
            return expr["value"] in self.variables
        else:
            return False

    def reduce(self, expr: dict) -> Union[float, np.ndarray]:
        if expr["type"] in "+-*/^":
            lhs = self.reduce(expr["left"])
            rhs = self.reduce(expr["right"])
            op = self.operations[expr["type"]]
            return op(lhs, rhs)
        elif expr["type"] == "num":
            try:
                return float(expr["value"])
            except ValueError:
                raise FormulaException(f"Not a number '{expr['value']}'.")
        elif expr["type"] == "var":
            try:
                return self.variables[expr["value"]]
            except KeyError:
                raise FormulaException(f"Unknown variable '{expr['value']}'.")
        else:
            raise FormulaException(f"Unknown expression type.")

    def validateString(self, string: str) -> bool:
        try:
            expr = self.parse(string)
        except FormulaException:
            return False
        return self.validate(expr)

    def reduceString(self, string: str) -> Union[float, np.ndarray]:
        return self.reduce(self.parse(string))
