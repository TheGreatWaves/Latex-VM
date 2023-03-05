from typing import List

import pytest
from _pytest.capture import CaptureFixture

from expression import Expression, pack, unpack
from graph_session import GraphSession


@pytest.mark.parametrize(
    "input",
    [r"**", r"\frac{2}{2", r"\frac2}{2"],
)
def test_try_parsing_fail(input: str):
    with pytest.raises(Exception):
        Expression.parse(input=input)


@pytest.mark.parametrize(
    "input",
    [
        "hello(('bob'), lol",
        "hello(2, 3",
    ],
)
def test_get_param_from_function_fail(input: str):
    with pytest.raises(Exception):
        Expression.get_parameters_str_from_function(input)


@pytest.mark.parametrize(
    "input",
    [
        "hello(('bob'), lol)",
        "hello(2, 3)",
    ],
)
def test_get_param_from_function(input: str):
    param = Expression.get_parameters_str_from_function(input)
    assert param == input[input.index("(") :]  # noqa: E203


def test_self_referential(gs: GraphSession):
    gs.execute("x=2")
    gs.execute("x=x+2")
    gs.execute("x=x")
    assert gs.get_session_variables()["x"] == "4"


def test_raw_expr(gs: GraphSession):
    gs.execute("x=2")
    gs.execute("x")
    assert gs.get_session_variables()["x"] == "2"


@pytest.mark.parametrize(
    "input",
    ["(2", "2", "((2)"],
)
def test_unpack_error(input: str):
    with pytest.raises(Exception):
        unpack(input)


@pytest.mark.parametrize(
    "input, exp",
    [("2", "(2)"), ("hello", "(hello)"), ("(x*y)", "((x*y))")],
)
def test_pack(input: str, exp: str):
    pack(input) == exp


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
    assert gs.get_session_variables()[lhs] == rhs


@pytest.mark.parametrize(
    "commands, exp",
    [
        ([r"f(x) = \frac{x}{2}", r"ans = f(1)"], "0.5"),
        ([r"f(x) = 2^{x}", r"ans = f(2)"], "4"),
        ([r"f(x) = 2^{x}", r"g(x) = 2^{f(x)}", r"ans = g(2)"], "16"),
        ([r"f(x) = 2^{x}", r"f(x) = f(x)+2", r"ans = f(2)"], "6"),
        ([r"x = 2^{2}", r"f(z) = z", r"f(y) = f(x)+y", r"ans = f(2)"], "6"),
        ([r"x = 2^{2}", r"f(z) = x + z", r"f(y) = f(x)+y", r"ans = f(2)"], "10"),
        ([r"f(x) = x^{2}", r"ans = f(2)"], "4"),
        ([r"f(x) = x^{2}", r"ans = f(2)"], "4"),
        (
            [
                r"long_function_name(some_var) = some_var",
                r"g(x) = long_function_name(x) * x",
                r"ans = g(2)",
            ],
            "4",
        ),
    ],
)
def test_function_declaration(gs: GraphSession, commands: List[str], exp: str):
    for command in commands:
        gs.execute(command)

    assert gs.get_session_variables()["ans"] == exp


def test_something(gs: GraphSession):
    gs.execute(r"x = 2^{2}")
    gs.execute(r"f(z) = z")
    gs.execute(r"f(y) = f(x)+y")
    gs.execute(r"ans = f(2)")
    assert gs.get_env_variables()["ans"] == "6"


@pytest.mark.parametrize(
    "commands, exp",
    [
        ([r"f(x) = x^2", "ans = f(f(2))"], "16"),
        ([r"f(x) = x^2", "f(x, y) = f(x) + f(y)", "ans = f(2, 2)"], "8"),
    ],
)
def test_nested_function(gs: GraphSession, commands: List[str], exp: str):
    for command in commands:
        gs.execute(command)
    assert gs.get_env_variables()["ans"] == exp


def test_stmt(gs: GraphSession, capfd: CaptureFixture[str]):
    gs.execute(r"f(x) = x^{2}")
    gs.execute(r"f(f(2))")
    out, _ = capfd.readouterr()
    assert out.endswith("16\n")
