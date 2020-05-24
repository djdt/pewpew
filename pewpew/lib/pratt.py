import numpy as np
import re

from typing import Dict, List, Union


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

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        raise ParserException("Invalid token.")


class Parens(Null):
    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Mismatched parenthesis.")
        return expr


class Value(Null):
    def __init__(self, value: str):
        self.value = value

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        return Expr(self.value)


class NaN(Null):
    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        return Expr("nan")


class Unary(Null):
    def __init__(self, value: str, rbp: int):
        self.value = value
        self.rbp = rbp

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        expr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[expr])


class Binary(Null):
    def __init__(self, value: str, div: str, rbp: int):
        self.value = value
        self.div = div
        self.rbp = rbp

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[expr, rexpr])


class Ternary(Null):
    def __init__(self, value: str, div: str, div2: str, rbp: int):
        self.value = value
        self.div = div
        self.div2 = div2
        self.rbp = rbp

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        lexpr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div2:
            raise ParserException(f"Missing '{self.div2}' statement.")
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[lexpr, expr, rexpr])


class UnaryFunction(Unary):
    def __init__(self, value: str):
        super().__init__(value, 0)

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


class BinaryFunction(Binary):
    def __init__(self, value: str):
        super().__init__(value, ",", 0)

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


class TernaryFunction(Ternary):
    def __init__(self, value: str):
        super().__init__(value, ",", ",", 0)

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


# Left Commands
class Left(object):
    lbp = -1

    @property
    def rbp(self):
        return self.lbp + 1

    def led(self, parser: "Parser", tokens: List[str], expr: Expr) -> Expr:
        raise ParserException("Invalid token.")


class LeftBinary(Left):
    def __init__(self, value: str, lbp: int, right: bool = False):
        self.value = value
        self.lbp = lbp
        self.right = right

    @property
    def rbp(self):
        return self.lbp + (0 if self.right else 1)

    def led(self, parser: "Parser", tokens: List[str], expr: Expr) -> Expr:
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[expr, rexpr])


class LeftTernary(Left):
    def __init__(self, value: str, div: str, lbp: int):
        self.value = value
        self.div = div
        self.lbp = lbp

    def led(self, parser: "Parser", tokens: List[str], lexpr: Expr) -> Expr:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(value=self.value, children=[lexpr, expr, rexpr])


class LeftIndex(Left):
    def __init__(self, value: str, lbp: int):
        self.value = value
        self.lbp = lbp

    def led(self, parser: "Parser", tokens: List[str], expr: Expr) -> Expr:
        rexpr = parser.parseExpr(tokens, 0)
        if len(tokens) == 0 or tokens.pop(0) != "]":
            raise ParserException("Mismatched bracket ']'.")
        return Expr(self.value, children=[expr, rexpr])


class Parser(object):
    def __init__(self, variables: List[str] = None):
        number_token = "\\d+\\.?\\d*(?:[eE][+\\-]?\\d+)?"
        operator_token = "[+\\-\\*/^!=<>?:]+"
        variable_token = "\\d*[a-zA-Z][a-zA-Z0-9_\\-]*"  # also covers if then else

        self.regexp_number = re.compile(number_token)
        self.regexp_tokenise = re.compile(
            f"\\s*([\\[\\]\\(\\)\\,]|{variable_token}|{number_token}|{operator_token})\\s*"
        )

        self.variables: List[str] = []
        if variables is not None:
            self.variables.extend(variables)

        self.nulls: Dict[str, Null] = {
            "(": Parens(),
            "if": Ternary("?", "then", "else", 11),
            "nan": NaN(),
            "-": Unary("u-", 30),
        }
        self.lefts: Dict[str, Left] = {
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
            "[": LeftIndex("[", 80),
        }

    def getNull(self, token: str) -> Null:
        if token in self.nulls:
            return self.nulls[token]
        if token in self.variables or self.regexp_number.fullmatch(token) is not None:
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
        self.variables: Dict[str, Union[float, np.ndarray]] = {}
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
            "[": (None, 2),
        }

    def reduceExpr(self, tokens: List[str]) -> Union[float, np.ndarray]:
        if len(tokens) == 0:
            raise ReducerException("Unexpected end of input.")
        token = tokens.pop(0)
        if token == "[":
            try:
                n = np.array(self.reduceExpr(tokens))
                i = self.reduceExpr(tokens)
                return n[int(i)]
            except (IndexError, TypeError, ValueError):
                raise ReducerException(f"Invalid indexing of '{n}' using '{i}'.")
        elif token in self.operations:
            op, nargs = self.operations[token]
            args = [self.reduceExpr(tokens) for i in range(nargs)]
            return op(*args)
        elif token in self.variables:
            return self.variables[token]
        else:
            try:
                if any(t in token for t in [".", "e", "E", "n"]):
                    return float(token)
                else:
                    return int(token)
            except ValueError:
                raise ReducerException(f"Unexpected input '{token}'.")

    def reduce(self, string: str) -> Union[float, np.ndarray]:
        tokens = string.split(" ")
        result = self.reduceExpr(tokens)
        if len(tokens) != 0:
            raise ReducerException(f"Unexpected input '{tokens[0]}'.")
        return result
