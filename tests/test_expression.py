from time import sleep
from typing import List

import pytest
from _pytest.capture import CaptureFixture

from src.expression import (
    Expression,
    pack,
    try_running,
    try_simplify_expression,
    unpack,
)
from src.graph_session import GraphSession
from src.type_defs import EnvironmentVariables


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
    "input, exp",
    [("2", "(2)"), ("hello", "(hello)"), ("(x*y)", "((x*y))")],
)
def test_unpack(input: str, exp: str):
    unpack(exp) == input


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


def test_long_param_names(gs: GraphSession):
    gs.execute(
        "f(some_long_var_name, some_long_var_name_2) = some_long_var_name * some_long_var_name_2"
    )

    if (f := gs.get_session_functions().get("f")) is not None:
        param, definition = f

        assert len(param) == 2
        assert param == ["some_long_varname", "some_long_var_name_2"]
        assert definition == "some_long_var_name name_long_var_name_2"


def test_env_variable_overwrite(gs: GraphSession):
    gs.execute("x = 5")
    gs.execute("y = 5")

    # X overwritten here
    gs.execute("x = 5 + y")
    gs.execute("double(number) = 2 * number")

    # X overwritten again
    gs.execute("x = double(x)")

    assert gs.get_env_variables() == {"x": "20", "y": "5"}


def test_use_of_existing_variable(gs: GraphSession):
    gs.execute("x = 2")
    gs.execute("y = 5")

    # grabs x from the environment to be used in the function
    gs.execute("double(n) = n x")
    gs.execute("var1 = double(5)")
    assert gs.get_session_variables().get("var1") == "10"

    gs.execute("mult(n, x) = n x")
    gs.execute("var2 = mult(5, 2)")
    assert gs.get_session_variables().get("var2") == "10"

    gs.execute("mult(n, x) = n x")
    gs.execute("var3 = mult(5, x)")
    assert gs.get_session_variables().get("var3") == "10"


def test_func_name_absolute(gs: GraphSession):
    gs.execute("f(x) = x")
    gs.execute("some_f(x) = x x")

    gs.execute("v1 = f(2)")
    gs.execute("v2 = some_f(2)")

    env_vars: EnvironmentVariables = gs.get_session_variables()
    assert env_vars is not None

    assert env_vars.get("v1") == "2"
    assert env_vars.get("v2") == "4"


def test_deeply_nested(gs: GraphSession):
    gs.execute("f(x, y) = x + y")
    gs.execute("some_f(x) = x x")
    gs.execute("some_other_f(x) = x*3")
    gs.execute(r"v1 = f(some_other_f(some_f(2)), \frac{some_other_f(some_f(2))}{2})")

    env_vars: EnvironmentVariables = gs.get_session_variables()
    assert env_vars.get("v1") == "18.0"


def test_empty_execution(gs: GraphSession):
    assert len(gs.get_session_variables()) == 0
    gs.execute("")
    assert len(gs.get_session_variables()) == 0


def test_break_expression():
    _, rhs = Expression.break_expression(raw_expr="some_f(1) + 2")
    assert rhs == ""


def test_try_running():
    def inf_loop():
        while True:
            pass

    res = try_running(inf_loop, 1.0)
    assert res is None

    def delayed_return() -> int:
        sleep(1)
        return 2

    res1 = try_running(delayed_return, 0.5)
    assert res1 is None

    res2 = try_running(delayed_return, 3)
    assert res2 == 2


def test_simplifying_statement_expression():
    short_equation: str = "2 + 2 + 2 + 2"
    simplified_short_equation: str = try_simplify_expression(short_equation)
    assert simplified_short_equation == "8"

    long_equation: str = r"g\left(x,\ y\right)\ =\frac{w\left(\sqrt{y\ ^{\frac{\sqrt{\frac{f\left(w\left(x\right)\right)\cdot2\ +\ y}{\sqrt{w\left(f\left(x\right)+w\left(2\right)\right)}\cdot3}}}{w\left(24\right)}}}\right)}{2\ \cdot\ \ln\ 2}"
    simplified_long_equation: str = try_simplify_expression(long_equation)
    assert simplified_long_equation == long_equation


def test_simplifying_function_expression():
    short_function_equation: str = "f(x) = 2 + 2 + 2 + 2 + x"
    simplified_function_short_equation: str = try_simplify_expression(
        short_function_equation
    )
    assert simplified_function_short_equation == "f(x) = x + 8"

    short_function_equation: str = "f(x, y) = x + x + x + y"
    simplified_function_short_equation: str = try_simplify_expression(
        short_function_equation
    )
    assert simplified_function_short_equation == "f(x, y) = 3 x + y"


def test_simplifying_assignment_expression():
    short_equation: str = "v = 2 + 2 + 2 + 2"
    simplified_short_equation: str = try_simplify_expression(short_equation)
    assert simplified_short_equation == "v = 8"
