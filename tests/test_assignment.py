from typing import List

import pytest

from latexvm.expression import Expression
from latexvm.graph_session import GraphSession


def test_assignment_error(gs: GraphSession):
    res = gs.execute(r"wowowowow = 5 = 2")
    assert str(res.message) == "Chaining assignment is not allowed"

    res = gs.execute(r"wowowowow = (2")
    assert "missing ')'" in str(res.message)


def test_invalid_assignment_lhs(gs: GraphSession):
    res = gs.execute("2 = 4")
    assert not res.ok() and "Invalid identifier" in str(res.message)

    res = gs.execute("x + x = x*2")
    assert not res.ok() and "Invalid assignment lhs" in str(res.message)

    res = gs.execute("x x = x*2")
    assert not res.ok() and "Invalid assignment lhs" in str(res.message)


def test_assignment_fail(gs: GraphSession):
    res = gs.execute(r"x = y")

    assert "Unresolved variable(s) found" in str(res.message)


@pytest.mark.parametrize(
    "commands, exp",
    [
        (["x = 5", "v = x", "v = x*2"], "v = 10"),
        (["x = 2", "x = x*x", "x = x*x"], "x = 16"),
        (["x = 2", "y = x*x", "h = x+y"], "h = 6"),
        ([r"x = \frac{1}{2}", r"y = \frac{x}{2}", r"h = \frac{y}{2}"], "h = 0.125"),
        ([r"x = \frac{1}{2}", r"y = x^2", r"h = y*4"], "h = 1.0"),
    ],
)
def test_var_chaining(gs: GraphSession, commands: List[str], exp: str):
    for command in commands:
        gs.execute(command)

    lhs, rhs = Expression.break_expression(exp)
    assert gs.get_env()[lhs] == rhs


@pytest.mark.parametrize(
    "exc, exp",
    [
        ("v = 5", "5"),
        ("whatwhat = 123123", "123123"),
        ("hello = 2", "2"),
    ],
)
def test_basic_var_declaration(gs: GraphSession, exc: str, exp: str):
    gs.execute(exc)
    assert gs.get_env_variables()[exc.split("=")[0].strip()] == exp
