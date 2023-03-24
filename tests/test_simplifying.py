from latexvm.expression import Expression, ExpressionBuffer
from latexvm.graph_session import GraphSession


def test_simplifying_statement_expression():
    short_equation: ExpressionBuffer = ExpressionBuffer.new("2 + 2 + 2 + 2")
    _ = Expression.try_simplify_expression(short_equation)
    assert short_equation.assemble() == "8"

    long_equation_str = r"g\left(x,\ y\right) = \frac{w\left(\sqrt{y\ ^{\frac{\sqrt{\frac{f\left(w\left(x\right)\right)\cdot2\ +\ y}{\sqrt{w\left(f\left(x\right)+w\left(2\right)\right)}\cdot3}}}{w\left(24\right)}}}\right)}{2\ \cdot\ \ln\ 2}"
    long_equation_str = Expression.replace_latex_parens(long_equation_str)
    print(f"eq: {long_equation_str}")
    long_equation: ExpressionBuffer = ExpressionBuffer.new(long_equation_str)
    _ = Expression.try_simplify_expression(long_equation)

    what = ExpressionBuffer.new(
        r"\frac{w\left(\sqrt{y\ ^{\frac{\sqrt{\frac{f\left(w\left(x\right)\right)\cdot2\ +\ y}{\sqrt{w\left(f\left(x\right)+w\left(2\right)\right)}\cdot3}}}{w\left(24\right)}}}\right)}{2\ \cdot\ \ln\ 2}"
    )
    Expression.try_simplify_expression(what)

    print(f"what: {what.assemble()}")

    print("WHY")
    print(long_equation.assemble())
    print(long_equation_str)
    assert long_equation.assemble() == long_equation_str


def test_simplifying_function_expression():
    short_function_equation: ExpressionBuffer = ExpressionBuffer.new(
        "f(x) = 2 + 2 + 2 + 2 + x"
    )
    Expression.try_simplify_expression(short_function_equation)
    assert short_function_equation.assemble() == "f(x) = x + 8"

    short_function_equation: ExpressionBuffer = ExpressionBuffer.new(
        "f(x, y) = x + x + x + y"
    )
    _ = Expression.try_simplify_expression(short_function_equation)
    assert short_function_equation.assemble() == "f(x, y) = 3 x + y"


def test_simplifying_assignment_expression():
    short_equation: ExpressionBuffer = ExpressionBuffer.new("v = 2 + 2 + 2 + 2")
    _ = Expression.try_simplify_expression(short_equation)
    assert short_equation.assemble() == "v = 8"


def test_enabling_simplify(gs: GraphSession):
    gs.execute("f(x) = x + x + x + x")
    assert gs.get_env_functions().get("f_func")[1] == "x + x + x + x"

    gs.execute("f(x) = x + x + x + x", True)
    assert gs.get_env_functions().get("f_func")[1] == "4 x"
