from time import sleep

import pytest

from latexvm.expression import Expression, ExpressionType
from latexvm.graph_session import GraphSession


@pytest.mark.parametrize(
    "input",
    [r"**", r"\frac{2}{2", r"\frac2}{2"],
)
def test_try_parsing_fail(input: str):
    with pytest.raises(Exception):
        Expression.parse(input=input)


def test_self_referential(gs: GraphSession):
    gs.execute("x=2")
    gs.execute("x=x+2")
    gs.execute("x=x")
    assert gs.get_env()["x"] == "4"


def test_raw_expr(gs: GraphSession):
    gs.execute("x=2")
    gs.execute("x")
    assert gs.get_env()["x"] == "2"


def test_break_expression():
    _, rhs = Expression.break_expression(raw_expr="some_f(1) + 2")
    assert rhs == ""


def test_try_running():
    def inf_loop():
        while True:
            pass

    res = Expression.try_running(inf_loop, 1.0)
    assert res is None

    def delayed_return() -> int:
        sleep(1)
        return 2

    res1 = Expression.try_running(delayed_return, 0.5)
    assert res1 is None

    res2 = Expression.try_running(delayed_return, 3)
    assert res2 == 2


def test_get_expr_type():
    assert (
        Expression.get_expression_type(r"\sum_{i=1}^{3}i") == ExpressionType.STATEMENT
    )
    assert (
        Expression.get_expression_type(r"v = \sum_{i=1}^{3}i")
        == ExpressionType.ASSIGNMENT
    )
    assert (
        Expression.get_expression_type(r"f(x) = \sum_{i=1}^{3}{i*x}")
        == ExpressionType.FUNCTION
    )
