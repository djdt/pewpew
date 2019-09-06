import numpy as np
import re

from typing import List, Union


class ParserException(Exception):
    pass


class Expr(object):
    def __init__(self, value: str, children: list = None):
        self.value = value
        self.children = children

    def __str__(self) -> str:
        if self.children is None:
            return self.value
        else:
            return f"( {self.value} {' '.join([str(c) for c in self.children])} )"


# Null Commands
class Null(object):
    rbp = -1

    def nud(self, parser: "Parser", tokens: List[str]) -> dict:
        raise NotImplementedError


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


# Left Commands
class Left(object):
    lbp = -1

    def led(self, parser: "Parser", tokens: List[str], expr: dict) -> dict:
        raise NotImplementedError


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
    def __init__(self, variables: List[str]):
        number_token = "\\d+\\.?\\d*(?:[eE]\\d+)?"
        operator_token = "[\\+\\-\\*\\/\\^\\!\\=\\<\\>\\?\\:]+"
        variable_token = "\\d*[a-zA-Z][a-zA-Z0-9_\\-]*"  # also covers if then else

        self.regexp_number = re.compile(number_token)
        self.regexp_tokenise = re.compile(
            f"\\s*([\\(\\)\\,]|{number_token}|{operator_token}|{variable_token})\\s*"
        )

        self.variables = variables
        self.nulls = {}
        self.lefts = {}

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

    def parse(self, string: str) -> dict:
        tokens = self.regexp_tokenise.findall(string)
        result = self.parseExpr(tokens)
        if len(tokens) != 0:
            raise ParserException(f"Unexpected input '{tokens[0]}'.")
        return result


class Reducer(object):
    def __init__(self, variables: dict):
        self.operations = {}
        self.variables = variables

    def reduceExpr(self, expr: Expr) -> Union[float, np.ndarray]:
        if expr.children is None:
            if expr.value in self.variables:
                return self.variables[expr.value]
            else:
                try:
                    return float(expr.value)
                except ValueError:
                    raise ParserException(f"Not a number '{expr.value}'.")
        elif expr.value in self.operations:
            op = self.operations[expr.value]
            args = [self.reduceExpr(c) for c in expr.children]
            return op(*args)
        else:
            raise ParserException(f"Unknown value {expr.value}.")


if __name__ == "__main__":

    basic_arith_null = {
        "(": Parens(),
        "if": Ternary("?", "then", "else"),
        "-": Unary("u-", 30),
    }
    basic_arith_left = {
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
    var = {"a": np.random.random((3, 3)), "b": np.arange(9).reshape(3, 3)}

    parser = Parser(list(var.keys()))
    parser.nulls.update(basic_arith_null)
    parser.lefts.update(basic_arith_left)

    lex = parser.parse("if b > 0.5 then 1 else b * 100 ^ 2")

    reducer = Reducer(var)
    reducer.operations.update(
        {
            "u-": np.negative,
            "+": np.add,
            "-": np.subtract,
            "*": np.multiply,
            "/": np.divide,
            "^": np.power,
            ">": np.greater,
            ">=": np.greater_equal,
            "<": np.less,
            "<=": np.less_equal,
            "=": np.equal,
            "!=": np.not_equal,
            "?": np.where,
        }
    )
