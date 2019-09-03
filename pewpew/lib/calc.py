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
        boolean_token = "[\\>\\<]\\=?|\\!?\\="
        number_token = "\\d+\\.?\\d*"
        operator_token = "[\\^\\*\\/\\+\\-]"
        variable_token = "\\d*[a-zA-Z]+\\d*"

        self.regexp_isbool = re.compile(boolean_token)
        self.regexp_isnum = re.compile(number_token)
        self.regexp_isop = re.compile(operator_token)
        self.regexp_isvar = re.compile(variable_token)
        self.regexp_tokenise = re.compile(
            f"\\s*([\\(\\)]|{boolean_token}|{number_token}|{operator_token}|{variable_token})\\s*"
        )

        self.booleans = {
            "<": np.less,
            "<=": np.less_equal,
            ">": np.greater,
            ">=": np.greater_equal,
            "=": np.equal,
            "!=": np.not_equal,
        }

        self.operations = {
            "+": np.add,
            "-": np.subtract,
            "*": np.multiply,
            "/": np.divide,
            "^": np.power,
        }

        self.variables = variables

    def isBoolean(self, string: str) -> bool:
        return self.regexp_isnum.match(string) is not None

    def isNumber(self, string: str) -> bool:
        return self.regexp_isnum.match(string) is not None

    def isOperator(self, string: str) -> bool:
        return self.regexp_isop.match(string) is not None

    def isVariable(self, string: str) -> bool:
        return self.regexp_isvar.match(string) is not None

    def parseOperators(self, tokens: List[str], ops: List[str]) -> dict:
        if len(ops) == 0:
            return self.parseType(tokens)

        expr = self.parseOperators(tokens, ops[1:])
        while len(tokens) > 0 and tokens[0] == ops[0]:
            token = tokens.pop(0)
            rhs = self.parseOperators(tokens, ops[1:])
            expr = dict(type="op", value=token, left=expr, right=rhs)
        return expr

    def parseType(self, tokens: List[str]) -> dict:
        if len(tokens) == 0:
            raise FormulaException(f"Unexpected end to input.")
        t = tokens.pop(0)
        if self.isNumber(t):
            return dict(type="num", value=t)
        elif self.isVariable(t):
            return dict(type="var", value=t)
        elif t == "(":
            expr = self.parseOperators(tokens, list(self.operations))
            if len(tokens) == 0 or tokens[0] != ")":
                raise FormulaException("Mismatched parenthesis.")
            tokens.pop(0)
            return expr
        else:
            raise FormulaException(f"Unexpected input '{t}'.")

    def parse(self, string: str) -> dict:
        tokens = self.regexp_tokenise.findall(string)

        ops = list(self.booleans.keys()) + list(self.operations.keys())
        result = self.parseOperators(tokens, ops)

        if len(tokens) != 0:
            raise FormulaException(f"Unexpected input '{tokens[0]}'.")

        return result

    def validate(self, expr: dict) -> bool:
        if expr["type"] == "op":
            return self.validate(expr["left"]) and self.validate(expr["right"])
        elif expr["type"] == "num":
            return True
        elif expr["type"] == "var":
            return expr["value"] in self.variables
        else:
            return False

    def reduce(self, expr: dict) -> Union[float, np.ndarray]:
        if expr["type"] == "op":
            lhs = self.reduce(expr["left"])
            rhs = self.reduce(expr["right"])
            op = self.operations[expr["value"]]
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


p = FormulaParser({"a": np.random.random((3, 3))})
# print(p.parse("1 + 2 <= 3 ^ 4"))
print(p.reduceString("(1 + 2 + 3) ^ 4 - a"))
