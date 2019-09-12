import numpy as np
import re

from typing import List, Union


class ParserException(Exception):
    pass


class ReducerException(Exception):
    pass


class Expr(object):
    def __init__(self, value: str, children: list = None):
        self.value = value
        self.children = children

    def __str__(self) -> str:
        if self.children is None:
            return self.value
        else:
            return f"{self.value} {' '.join([str(c) for c in self.children])}"


# Null Commands
class Null(object):
    rbp = -1

    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        raise ParserException("Invalid token.")


class Parens(Null):
    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Mismatched parenthesis.")
        return expr


class Value(Null):
    def __init__(self, value: str):
        self.value = value

    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        return Expr(self.value)


class NaN(Null):
    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        return Expr("nan")


class Unary(Null):
    def __init__(self, value: str, rbp: int):
        self.value = value
        self.rbp = rbp

    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        expr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[expr])


class Binary(Null):
    def __init__(self, value: str, div: str):
        self.value = value
        self.div = div

    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        rexpr = parser.parseExpr(tokens)
        return Expr(self.value, children=[expr, rexpr])


class Ternary(Null):
    def __init__(self, value: str, div: str, div2: str):
        self.value = value
        self.div = div
        self.div2 = div2

    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        lexpr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div2:
            raise ParserException(f"Missing '{self.div2}' statement.")
        rexpr = parser.parseExpr(tokens)
        return Expr(self.value, children=[lexpr, expr, rexpr])


class UnaryFunction(Unary):
    def __init__(self, value: str):
        super().__init__(value, 0)

    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


class BinaryFunction(Binary):
    def __init__(self, value: str):
        super().__init__(value, ",")

    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


class TernaryFunction(Ternary):
    def __init__(self, value: str):
        super().__init__(value, ",", ",")

    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


# Left Commands
class Left(object):
    lbp = -1

    def led(self, parser: "Parser", tokens: List[str], expr: dict) -> dict:
        raise ParserException("Invalid token.")


class LeftBinary(Left):
    def __init__(self, value: str, lbp: int, right: bool = False):
        self.value = value
        self.lbp = lbp
        self.right = right

    @property
    def rbp(self):
        return self.lbp + (0 if self.right else 1)

    def led(self, parser: "Parser", tokens: List[str], expr: dict) -> dict:
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[expr, rexpr])


class LeftTernary(Left):
    def __init__(self, value: str, div: str, lbp: int):
        self.value = value
        self.div = div
        self.lbp = lbp

    @property
    def rbp(self):
        return self.lbp + 1

    def led(self, parser: "Parser", tokens: List[str], lexpr: dict) -> dict:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(value=self.value, children=[lexpr, expr, rexpr])


class Parser(object):
    def __init__(self, variables: List[str] = None):
        number_token = "\\d+\\.?\\d*(?:[eE][+\\-]?\\d+)?"
        operator_token = "[+\\-\\*/^!=<>?:]+"
        variable_token = "\\d*[a-zA-Z][a-zA-Z0-9_\\-]*"  # also covers if then else

        self.regexp_number = re.compile(number_token)
        self.regexp_tokenise = re.compile(
            f"\\s*([\\(\\)\\,]|{variable_token}|{number_token}|{operator_token})\\s*"
        )

        self.variables = []
        if variables is not None:
            self.variables.extend(variables)

        self.nulls = {
            "(": Parens(),
            "if": Ternary("?", "then", "else"),
            "nan": NaN(),
            "-": Unary("u-", 30),
        }
        self.lefts = {
            "?": LeftTernary("?", ":", 10),
            "<": LeftBinary("<", 10),
            "<=": LeftBinary("<=", 10),
            ">": LeftBinary(">", 10),
            ">=": LeftBinary(">=", 10),
            "=": LeftBinary("=", 10),
            "!=": LeftBinary("!=", 10),
            "+": LeftBinary("+", 20),
            "-": LeftBinary("-", 20),
            "*": LeftBinary("*", 40),
            "/": LeftBinary("/", 40),
            "^": LeftBinary("^", 50, right=True),
        }

    def getNull(self, token: str) -> Null:
        if token in self.nulls:
            return self.nulls[token]
        if token in self.variables or self.regexp_number.match(token) is not None:
            return Value(token)
        return Null()

    def getLeft(self, token: str) -> Left:
        if token in self.lefts:
            return self.lefts[token]
        return Left()

    def parseExpr(self, tokens: List[str], prec: int = 0) -> Expr:
        if len(tokens) == 0:
            raise ParserException("Unexpected end of input.")

        token = tokens.pop(0)
        cmd = self.getNull(token)
        expr = cmd.nud(self, tokens)
        while len(tokens) > 0:
            lcmd = self.getLeft(tokens[0])
            if prec > lcmd.lbp:
                break
            tokens.pop(0)
            expr = lcmd.led(self, tokens, expr)
        return expr

    def parse(self, string: str) -> str:
        tokens = self.regexp_tokenise.findall(string)
        result = self.parseExpr(tokens)
        if len(tokens) != 0:
            raise ParserException(f"Unexpected input '{tokens[0]}'.")
        return str(result)


class Reducer(object):
    def __init__(self, variables: dict = None):
        self.variables = {"nan": np.nan}
        if variables is not None:
            self.variables.update(variables)

        self.operations = {
            "u-": (np.negative, 1),
            "+": (np.add, 2),
            "-": (np.subtract, 2),
            "*": (np.multiply, 2),
            "/": (np.divide, 2),
            "^": (np.power, 2),
            ">": (np.greater, 2),
            ">=": (np.greater_equal, 2),
            "<": (np.less, 2),
            "<=": (np.less_equal, 2),
            "=": (np.equal, 2),
            "!=": (np.not_equal, 2),
            "?": (np.where, 3),
        }

    def reduceExpr(self, tokens: List[str]) -> Union[float, np.ndarray]:
        if len(tokens) == 0:
            raise ReducerException("Unexpected end of input.")
        token = tokens.pop(0)
        if token in self.operations:
            op, nargs = self.operations[token]
            args = [self.reduceExpr(tokens) for i in range(nargs)]
            return op(*args)
        elif token in self.variables:
            return self.variables[token]
        else:
            try:
                return float(token)
            except ValueError:
                raise ReducerException(f"Unexpected input '{token}'.")

    def reduce(self, string: str) -> Union[float, np.ndarray]:
        tokens = string.split(" ")
        result = self.reduceExpr(tokens)
        if len(tokens) != 0:
            raise ReducerException(f"Unexpected input '{tokens[0]}'.")
        return result
