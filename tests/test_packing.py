import pytest

from latexvm.expression import Expression


@pytest.mark.parametrize(
    "input",
    ["(2", "2", "((2)"],
)
def test_unpack_error(input: str):
    with pytest.raises(Exception):
        Expression.unpack(input)


@pytest.mark.parametrize(
    "input, exp",
    [("2", "(2)"), ("hello", "(hello)"), ("(x*y)", "((x*y))")],
)
def test_pack(input: str, exp: str):
    Expression.pack(input) == exp


@pytest.mark.parametrize(
    "input, exp",
    [("2", "(2)"), ("hello", "(hello)"), ("(x*y)", "((x*y))")],
)
def test_unpack(input: str, exp: str):
    Expression.unpack(exp) == input
