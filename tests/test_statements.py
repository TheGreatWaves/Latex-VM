import pytest

from latexvm.graph_session import GraphSession


def test_statement(gs: GraphSession):
    gs.execute(r"f(x) = x^{2}")
    res = gs.execute(r"f(f(2))")

    assert res.ok() and "16" in res.message

    res1 = gs.execute(r"2")
    res2 = gs.execute(r"2.3")
    assert res1.ok() and "2.0" == res1.message
    assert res2.ok() and "2.3" == res2.message


@pytest.mark.parametrize(
    "input, expected",
    [
        (r"y", "Unresolved variable(s) found"),
        (r"f(x)", "Unresolved variable(s) found"),  # Variable is resolved first
        (r"f(5)", "Unresolved function(s) found"),
        (r"\what? 42!", "I don't understand"),
    ],
)
def test_statement_fail(gs: GraphSession, input: str, expected: str):
    res = gs.execute(input=input)
    assert expected in str(res.message)
