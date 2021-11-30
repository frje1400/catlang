import operator
import re

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterable

# Uses the verbose mode where whitespace in the pattern is ignored and comments are allowed.
token_patterns = re.compile(r"""
    -?[0-9]+                            |  # integer
    config|print|dec|hex|bin            |  # keywords
    \S                                  |  # non-space single character, e.g. ( and +
    [a-zA-Z][a-zA-Z0-9]*                |  # variables
""", re.VERBOSE)


class CatError(Exception):
    pass


class TType(Enum):
    integer = auto()
    keyword = auto()
    left_paren = auto()
    right_paren = auto()
    variable = auto()
    add = auto()
    sub = auto()
    mul = auto()
    div = auto()
    print = auto()
    config = auto()
    dec = auto()
    bin = auto()
    hex = auto()
    equals = auto()


# Maps from a token type to a math operator callable.
lookup_operator = {
    TType.add: operator.add,
    TType.sub: operator.sub,
    TType.mul: operator.mul,
    TType.div: operator.floordiv
}

# Simple tokens are tokens that consist of a single character.
simple_tokens = {
    "+": TType.add,
    "-": TType.sub,
    "*": TType.mul,
    "/": TType.div,
    "(": TType.left_paren,
    ")": TType.right_paren,
    "=": TType.equals
}

# The four valid keywords in cat.
keywords = {"config", "print", "bin", "dec", "hex"}


@dataclass
class Token:
    tok_type: TType
    lexeme: str


@dataclass
class Node:
    pass


@dataclass
class Statement(Node):
    pass


@dataclass
class Expression(Node):
    pass


@dataclass
class ConfigStmt(Statement):
    number_base: TType


@dataclass
class AssignStmt(Statement):
    var: Token
    expr: Expression


@dataclass
class PrintStmt(Statement):
    expr: Expression


@dataclass
class BinaryExpression(Expression):
    first: Expression
    operator: Token
    second: Expression


@dataclass
class PrimaryExpression(Expression):
    pass


@dataclass
class Integer(PrimaryExpression):
    integer: Token


@dataclass
class Variable(PrimaryExpression):
    variable: Token


@dataclass
class Paren(PrimaryExpression):
    exp: Expression


def tokenize(source: str) -> deque[Token]:
    """Splits the source code on the regex pattern that describes the valid tokens
    and categorizes the tokens by associating them with the TokenType enum."""
    tokens = [categorize_token(token) for token in token_patterns.findall(source) if len(token) > 0]
    return deque(tokens)


def categorize_token(token: str) -> Token:
    """Looks at a piece of cat source code and returns the matching token type."""
    match token:
        case token if token in simple_tokens:
            return Token(simple_tokens[token], token)
        case token if is_integer(token):
            return Token(TType.integer, token)
        case "print":
            return Token(TType.print, token)
        case "config":
            return Token(TType.config, token)
        case "bin":
            return Token(TType.bin, token)
        case "hex":
            return Token(TType.hex, token)
        case "dec":
            return Token(TType.dec, token)
        case token if is_variable(token):
            return Token(TType.variable, token)


def is_variable(token: str) -> bool:
    """Returns True if token is a valid Cat variable name."""
    return re.match(r"""[a-zA-Z][a-zA-Z0-9]*""", token) and token not in keywords


def is_integer(token: str) -> bool:
    """Return True if token is a Cat integer."""
    return re.match(r"""-?[0-9]+""", token) is not None


def pop_or_raise(tokens: deque[Token], message: str, *expected_tokens) -> Token:
    """Removes and returns a from the tokens queue if the token type of the
    first token is in the expected tokens, otherwise raises an error."""
    return _or_raise(tokens, message, lambda deq: deq.popleft(), *expected_tokens)


def peek_or_raise(tokens: deque[Token], message: str, *expected_tokens) -> Token:
    """Returns a token from the queue if the token type of the first token is
    in the expected tokens, otherwise raises an error."""
    return _or_raise(tokens, message, lambda deq: deq[0], *expected_tokens)


def _or_raise(tokens: deque[Token], message: str, deque_op: Callable, *expected_tokens) -> Token:
    if len(tokens) == 0:
        raise CatError(f"Unexpectedly ran out of tokens when trying to find:\n{expected_tokens}")

    token = deque_op(tokens)
    if any([t_type for t_type in expected_tokens if t_type == token.tok_type]):
        return token

    raise CatError(f"{message}\nFound: {token.lexeme}")


def peek_or_pass(tokens: deque[Token], *expected_tokens) -> bool:
    """Returns True if the queue contains token(s) and the first token
    is in the expected tokens."""
    return len(tokens) != 0 and any(tok_type for tok_type in expected_tokens
                                    if tok_type == tokens[0].tok_type)


def parse(tokens: deque[Token]) -> Iterable[Statement]:
    """Transforms a queue of tokens into an iterable of statements."""
    statements = []

    while len(tokens) > 0:
        statement = parse_statement(tokens)
        statements.append(statement)

    return statements


def parse_statement(tokens: deque[Token]) -> Statement:
    token = peek_or_raise(tokens, "Expected statement: config, print or variable.",
                          TType.config, TType.print, TType.variable)

    match token.tok_type:
        case TType.config:
            return parse_config(tokens)
        case TType.print:
            return parse_print(tokens)
        case TType.variable:
            return parse_assign(tokens)


def parse_config(tokens: deque[Token]) -> ConfigStmt:
    config = pop_or_raise(tokens, "Expected config.", TType.config)
    config_type = pop_or_raise(tokens, "Expected config type: dec, bin, hex.",
                               TType.bin, TType.dec, TType.hex)
    return ConfigStmt(config_type.tok_type)


def parse_assign(tokens: deque[Token]) -> AssignStmt:
    variable = pop_or_raise(tokens, "Expected variable.", TType.variable)
    equals = pop_or_raise(tokens, "Expected equals.", TType.equals)
    math_exp = parse_math_exp(tokens)
    return AssignStmt(variable, math_exp)


def parse_print(tokens: deque[Token]) -> PrintStmt:
    print_token = pop_or_raise(tokens, "Expected print.", TType.print)
    math_exp = parse_math_exp(tokens)
    return PrintStmt(math_exp)


def parse_math_exp(tokens: deque[Token]) -> Expression:
    return parse_sum_exp(tokens)


def parse_sum_exp(tokens: deque[Token]) -> Expression:
    exp = parse_product(tokens)

    while peek_or_pass(tokens, TType.add, TType.sub):
        op = pop_or_raise(tokens, "Expected + or -.", TType.add, TType.sub)
        second = parse_product(tokens)
        exp = BinaryExpression(exp, op, second)

    return exp


def parse_product(tokens: deque[Token]) -> Expression:
    exp = parse_primary(tokens)

    while peek_or_pass(tokens, TType.mul, TType.div):
        op = pop_or_raise(tokens, "Expected * or /.", TType.mul, TType.div)
        second = parse_primary(tokens)
        exp = BinaryExpression(exp, op, second)

    return exp


def parse_primary(tokens: deque[Token]) -> PrimaryExpression:
    token = pop_or_raise(tokens, "Expected primary: integer, variable or left paren.",
                         TType.integer, TType.variable, TType.left_paren)
    match token.tok_type:
        case TType.integer:
            return Integer(token)
        case TType.variable:
            return Variable(token)
        case TType.left_paren:
            math_exp = parse_math_exp(tokens)
            right_paren = pop_or_raise(tokens, "Expected right paren.", TType.right_paren)
            return Paren(math_exp)


def evaluate(statements: Iterable[Statement]) -> Iterable[str]:
    env = {}
    number_base = TType.dec

    for statement in statements:
        match statement:
            case PrintStmt(expr=expr):
                value = eval_exp(expr, env)
                printable_value = to_print(value, number_base)
                yield printable_value
            case AssignStmt(var=var, expr=expr):
                name = var.lexeme
                value = eval_exp(expr, env)
                env[name] = value
            case ConfigStmt(number_base=number_base):
                number_base = number_base
            case _:
                raise CatError(f"Unknown statement type: {statement}")


def to_print(value: int, number_base: TType) -> str:
    match number_base:
        case TType.dec:
            return str(value)
        case TType.bin:
            return bin(value)
        case TType.hex:
            return hex(value)


def eval_exp(node: dict, env: dict) -> int:
    match node:
        case Integer(integer=integer):
            return int(integer.lexeme)
        case BinaryExpression(first=first, operator=op, second=second):
            first_value = eval_exp(first, env)
            op = lookup_operator[op.tok_type]
            second_value = eval_exp(second, env)
            return op(first_value, second_value)
        case Paren(exp=expr):
            return eval_exp(expr, env)
        case Variable(variable=var):
            return env[var.lexeme]
        case _:
            raise CatError(f"Unknown node type: {node}")


def run_code(source_code: str) -> Iterable[str]:
    return evaluate(parse(tokenize(source_code)))
