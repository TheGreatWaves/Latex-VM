from typing import List

import pytest

from latexvm.expression import Expression
from latexvm.graph_session import GraphSession
from latexvm.type_defs import EnvironmentVariables


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


def test_invalid_arity(gs: GraphSession):
    gs.execute("y = 20")
    gs.execute(r"double(x) = x*(2)")
    gs.execute(r"f(x, y) = double(x) * y")

    input = r"\frac{2}{3} + f(\frac{2}{3})"
    res = gs.force_resolve_function(input)

    assert not res.ok() and "Function arity error" in str(res.message)


def test_invalid_function_lhs(gs: GraphSession):
    res = gs.execute("what(x = 4")
    assert "Invalid function lhs" in str(res.message)

    res = gs.execute("2(x) = x*2")
    assert "Invalid function lhs" in str(res.message)


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

    assert gs.get_env()["ans"] == exp


def test_function_calls(gs: GraphSession):
    gs.execute(r"x = 2^{2}")
    gs.execute(r"f(z) = z")
    gs.execute(r"f(y) = f(x)+y")
    gs.execute(r"ans = f(2)")
    assert gs.get_env_variables()["ans"] == "6"


def test_long_param_names(gs: GraphSession):
    gs.execute(
        "f(some_long_var_name, some_long_var_name_2) = some_long_var_name * some_long_var_name_2"
    )

    f = gs.get_env_functions().get("f")
    if f is not None:
        param, definition = f

        assert len(param) == 2
        assert param == ["some_long_varname", "some_long_var_name_2"]
        assert definition == "some_long_var_name name_long_var_name_2"


def test_deeply_nested(gs: GraphSession):
    gs.execute("f(x, y) = x + y")
    gs.execute("some_f(x) = x x")
    gs.execute("some_other_f(x) = x*3")
    gs.execute(r"v1 = f(some_other_f(some_f(2)), \frac{some_other_f(some_f(2))}{2})")

    env_vars: EnvironmentVariables = gs.get_env()
    assert env_vars.get("v1") == "18.0"


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
