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

number_token = "\\d+\\.?\\d*"
variable_token = "\\d*[a-zA-Z]+\\d*"
regexp_isvar = re.compile(variable_token)
regexp_isnum = re.compile(number_token)

regexp_tokenise = re.compile(f"\\s*({variable_token}|{number_token}|\\S)\\s*")


class FormulaException(Exception):
    pass


class FormulaParser(object):
    def __init__(self, variables: dict):
        self.allowed_operators = "+-*/^()"
        self.variables = variables

    def isVariable(self, string: str) -> bool:
        return regexp_isvar.match(string) is not None

    def isNumber(self, string: str) -> bool:
        return regexp_isnum.match(string) is not None

    def convert(self, string: str) -> Union[str, float, np.ndarray]:
        if self.isNumber(string):
            return float(string)
        elif self.isVariable(string):
            try:
                return np.array(self.variables[string], dtype=float)
            except KeyError:
                raise FormulaException("Unknown variable.")
        elif string in self.allowed_operators:
            return string
        else:
            raise FormulaException("Invalid input.")

    def tokenise(self, string: str):
        tokens = [self.convert(token) for token in regexp_tokenise.findall(string)]
        print(tokens)
        return tokens

    def findOuterParenthesis(self, tokens: List) -> np.ndarray:
        try:
            start = tokens.index("(")
            end = next(i for i in reversed(range(len(tokens))) if tokens[i] == ")")
        except (ValueError, StopIteration):
            raise FormulaException("Mismatched parenthesis.")
        return start, end

    def parse(self, tokens: List) -> np.ndarray:
        # 1. Look for any braces, if there are, look for outermost and recurse in
        if tokens.count("(") > 0:
            start, end = self.findOuterParenthesis(tokens)
            tokens = tokens[:start] + self.parse(tokens[start:end]) + tokens[end:]
        # 2. Apply ops, respecting order of ops
        for op in self.order_ops:



p = FormulaParser({"P31": np.random.random((10, 10))})
p.tokenise("1 + 2 * (11.001 + 1)")
p.tokenise("1 + P31 + 1")
# p.tokenise("1 + 1+2 * (P31 * 31P)")
