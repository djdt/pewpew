import numpy as np
import re

from typing import Callable, List, Union


class ParserException(Exception):
    pass


class Parser(object):
    def __init__(self, variables: dict):
        number_token = "\\d+\\.?\\d*(?:[eE]\\d+)?"
        operator_token = "[\\+\\-\\*\\/\\^\\!\\=\\<\\>]+"
        variable_token = "\\d*[a-zA-Z][a-zA-Z0-9_\\-]*"  # also covers if then else

        self.regexp_tokenise = re.compile(
            f"\\s*([\\(\\)]|{number_token}|{operator_token}|{variable_token})\\s*"
        )

        self.variables = variables

        self.nullcmds = {
            "(": Parens(),
            "if": IfThenElse(np.where),
            "-": UnaryOp("-", np.negative, 30),
        }
        self.leftcmds = {
            "<": BinaryOp("<", np.less, 10),
            "<=": BinaryOp("<=", np.less_equal, 10),
            ">": BinaryOp(">", np.greater, 10),
            ">=": BinaryOp(">=", np.greater_equal, 10),
            "=": BinaryOp("=", np.equal, 10),
            "!=": BinaryOp("!=", np.not_equal, 10),
            "+": BinaryOp("+", np.add, 20),
            "-": BinaryOp("-", np.subtract, 20),
            "*": BinaryOp("*", np.multiply, 30),
            "/": BinaryOp("/", np.divide, 30),
            "^": BinaryOp("^", np.power, 50, right=True),
        }

    def parseExpr(self, tokens: List[str], prec: int = 0) -> dict:
        if len(tokens) == 0:
            raise ParserException("Unexpected end of input.")

        token = tokens.pop(0)
        cmd = self.nullcmds.get(token, Value(token))
        expr = cmd.nud(self, tokens)
        while len(tokens) > 0:
            cmd = self.leftcmds.get(tokens[0], Left())
            if prec > cmd.lbp:
                break
            tokens.pop(0)
            expr = cmd.led(self, tokens, expr)
        return expr

    def parse(self, string: str) -> dict:
        tokens = self.regexp_tokenise.findall(string)
        result = self.parseExpr(tokens)
        if len(tokens) != 0:
            raise ParserException(f"Unexpected input '{tokens[0]}'.")
        return result

    def reduce(self, expr: dict) -> Union[float, np.ndarray]:
        if expr["type"] == "value":
            if expr["value"] in self.variables:
                return self.variables[expr["value"]]
            try:
                return float(expr["value"])
            except ValueError:
                raise ParserException(f"Unknown value '{expr['value']}'.")
        elif expr["type"] == "unary":
            a = self.reduce(expr["right"])
            op = self.nullcmds[expr["value"]]
            return op.func(a)
        elif expr["type"] == "binary":
            a = self.reduce(expr["left"])
            b = self.reduce(expr["right"])
            op = self.leftcmds[expr["value"]]
            return op.func(a, b)
        elif expr["type"] == "ternary":
            a = self.reduce(expr["left"])
            b = self.reduce(expr["center"])
            c = self.reduce(expr["right"])
            op = self.nullcmds[expr["value"]]
            return op.func(a, b, c)
        else:
            raise ParserException(f"Unknown expression type.")

    def reduceString(self, string: str) -> dict:
        return self.reduce(self.parse(string))


# Null Commands
class Null(object):
    rbp = -1

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        raise NotImplementedError


class Value(object):
    def __init__(self, value: str):
        self.value = value

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        return dict(type="value", value=self.value)


class UnaryOp(Null):
    def __init__(self, symbol: str, func: Callable, rbp: int):
        self.symbol = symbol
        self.func = func
        self.rbp = rbp

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        return dict(
            type="unary", value=self.symbol, right=parser.parseExpr(tokens, self.rbp)
        )


class Parens(Null):
    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Mismatched parenthesis.")
        return expr


class IfThenElse(Null):
    def __init__(self, func: Callable):
        self.func = func

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        lexpr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != "then":
            raise ParserException("Missing 'then' statement.")
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != "else":
            raise ParserException("Missing 'else' statement.")
        rexpr = parser.parseExpr(tokens)
        return dict(type="ternary", value="if", left=lexpr, center=expr, right=rexpr)


# Left Commands
class Left(object):
    lbp = -1

    def led(self, parser: Parser, tokens: List[str], expr: dict) -> dict:
        raise NotImplementedError


class BinaryOp(Left):
    def __init__(self, symbol: str, func: Callable, lbp: int, right: bool = False):
        self.symbol = symbol
        self.func = func
        self.lbp = lbp
        self.right = right

    @property
    def rbp(self):
        return self.lbp + (0 if self.right else 1)

    def led(self, parser: Parser, tokens: List[str], expr: dict) -> dict:
        rexpr = parser.parseExpr(tokens, self.rbp)
        return dict(type="binary", value=self.symbol, left=expr, right=rexpr)


class PostFix(BinaryOp):
    def led(self, parser: Parser, tokens: List[str], expr: dict) -> dict:
        return dict(type="postfix", left=expr)
