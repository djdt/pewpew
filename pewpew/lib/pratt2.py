import numpy as np
import re

from typing import Callable, List, Union


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


class Parser(object):
    def __init__(self, variables: dict):
        number_token = "\\d+\\.?\\d*(?:[eE]\\d+)?"
        operator_token = "[\\+\\-\\*\\/\\^\\!\\=\\<\\>\\?\\:]+"
        variable_token = "\\d*[a-zA-Z][a-zA-Z0-9_\\-]*"  # also covers if then else

        self.regexp_tokenise = re.compile(
            f"\\s*([\\(\\)\\,]|{number_token}|{operator_token}|{variable_token})\\s*"
        )

        self.variables = variables

        self.nullcmds = {
            "(": Parens(),
            "if": Ternary("if", "then", "else", np.where),
            "percentile": BinaryFunction("percentile", np.percentile),
            "-": Unary("-", np.negative, 30),
        }
        self.leftcmds = {
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

    def parseExpr(self, tokens: List[str], prec: int = 0) -> dict:
        if len(tokens) == 0:
            raise ParserException("Unexpected end of input.")

        token = tokens.pop(0)
        cmd = self.nullcmds.get(token, Value(token))
        expr = cmd.nud(self, tokens)
        while len(tokens) > 0:
            lcmd = self.leftcmds.get(tokens[0], Left())
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

    def reduce(self, expr: dict) -> Union[float, np.ndarray]:
        if expr["type"] == "value":
            if expr["value"] in self.variables:
                return self.variables[expr["value"]]
            try:
                return float(expr["value"])
            except ValueError:
                raise ParserException(f"Unknown value '{expr['value']}'.")
        elif expr["type"] == "null":
            args = [self.reduce(expr["a"])]
            if expr["order"] in ["binary", "ternary"]:
                args.append(self.reduce(expr["b"]))
            if expr["order"] == "ternary":
                args.append(self.reduce(expr["c"]))

            op: Union[Null, Left] = self.nullcmds[expr["value"]]
            if not isinstance(op, (Unary, Binary, Ternary)):
                raise ParserException("Invalid null token.")
            return op.func(*args)
        elif expr["type"] == "left":
            args = [self.reduce(expr["a"])]
            if expr["order"] in ["binary", "ternary"]:
                args.append(self.reduce(expr["b"]))
            if expr["order"] == "ternary":
                args.append(self.reduce(expr["c"]))

            op = self.leftcmds[expr["value"]]
            if not isinstance(op, (LeftBinary, LeftTernary)):
                raise ParserException("Invalid left token.")
            return op.func(*args)
        else:
            raise ParserException(f"Unknown expression type.")

    def reduceString(self, string: str) -> Union[float, np.ndarray]:
        return self.reduce(self.parse(string))


# Null Commands
class Null(object):
    rbp = -1

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        raise NotImplementedError


class Parens(Null):
    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Mismatched parenthesis.")
        return expr


class Value(object):
    def __init__(self, value: str):
        self.value = value

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        return Node(type="value", value=self.value)


class Unary(Null):
    def __init__(self, symbol: str, func: Callable, rbp: int):
        self.symbol = symbol
        self.func = func
        self.rbp = rbp

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        expr = parser.parseExpr(tokens, self.rbp)
        return Node(type="null", order="unary", value=self.symbol, a=expr)


class Binary(Null):
    def __init__(self, symbol: str, div: str, func: Callable):
        self.symbol = symbol
        self.func = func
        self.div = div

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        rexpr = parser.parseExpr(tokens)
        return dict(type="null", order="binary", value=self.symbol, a=expr, b=rexpr)


class Ternary(Null):
    def __init__(self, symbol: str, div: str, div2: str, func: Callable):
        self.symbol = symbol
        self.func = func
        self.div = div
        self.div2 = div2

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        lexpr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div2:
            raise ParserException(f"Missing '{self.div2}' statement.")
        rexpr = parser.parseExpr(tokens)
        return dict(
            type="null", order="ternary", value=self.symbol, a=lexpr, b=expr, c=rexpr
        )


class BinaryFunction(Binary):
    def __init__(self, symbol: str, func: Callable):
        super().__init__(symbol, ",", func)

    def nud(self, parser: Parser, tokens: List[str]) -> dict:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


# Left Commands
class Left(object):
    lbp = -1

    def led(self, parser: Parser, tokens: List[str], expr: dict) -> dict:
        raise NotImplementedError


class LeftBinary(Left):
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
        return dict(type="left", order="binary", value=self.symbol, a=expr, b=rexpr)


class LeftTernary(Left):
    def __init__(self, symbol: str, div: str, func: Callable, lbp: int):
        self.symbol = symbol
        self.func = func
        self.div = div
        self.lbp = lbp

    @property
    def rbp(self):
        return self.lbp + 1

    def led(self, parser: Parser, tokens: List[str], lexpr: dict) -> dict:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        rexpr = parser.parseExpr(tokens, self.rbp)
        return dict(
            type="left", order="ternary", value=self.symbol, a=lexpr, b=expr, c=rexpr
        )


if __name__ == "__main__":
    parser = Parser({"a": np.random.random((3, 3)), "b": np.arange(9).reshape(3, 3)})

    print(parser.reduceString("if a > 0.5 then a else b * 100 ^ 2"))
    print(parser.reduceString("a > 0.5 ? a : b * 100 ^ 2"))
