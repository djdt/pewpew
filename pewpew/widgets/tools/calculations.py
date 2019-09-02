import numpy as np
import re

# from PySide2 import QtCore, QtWidgets

# from laserlib.krisskross.data import krisskross_layers

# from pewpew.lib.viewoptions import ViewOptions
# from pewpew.validators import DecimalValidator
# from pewpew.widgets.canvases import LaserCanvas
# from pewpew.widgets.docks import LaserImageDock, KrissKrossImageDock

# from .tool import Tool

from typing import List, Union


# class CalulationsLineEdit(QtWidgets.QLineEdit):
#     OPERATORS = ["+", "-", "/", "*", "^"]

#     def __init__(self, istopes: List[str], parent: QtWidgets.QWidget = None):
#         super().__init__(parent)

#     def hasAcceptableInput(self):
#         return super().hasAcceptableInput()


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
        t = tokens.pop(0)
        if self.isNumber(t):
            return dict(type="num", value=t)
        elif self.isVariable(t):
            return dict(type="var", value=t)
        elif t == "(":
            expr = self.parseAddition(tokens)
            if tokens[0] != ")":
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


p = FormulaParser({"P31": np.random.random((10, 10))})
r1 = p.parse("1 + 2 * (11.001 + 1)")
r2 = p.parse("1 + P32 + 1")
