from latexvm.graph_session import GraphSession


def test_invalid_unresolved(gs: GraphSession):
    gs.execute("y = 20")
    gs.execute(r"double(x) = x*(2)")
    gs.execute(r"f(x, y) = double(x) * y")

    input = r"\frac{2}{3} + g(\frac{2}{3})"
    res = gs.force_resolve_function(input)
    print(res)

    assert not res.ok() and "Unresolved function(s) found" in str(res.message)


def test_force_resolve(gs: GraphSession):
    gs.execute("y = 20")
    gs.execute(r"double(x) = x*(2)")
    gs.execute(r"f(x, y) = double(x) * y")

    input = r"\frac{2}{3} + f(\frac{2}{3}, y)"
    res = gs.force_resolve_function(input)
    assert res.ok() and "2/3 + ((2/3)*2)*20" in str(res.message)


def test_arity_error(gs: GraphSession):
    gs.execute("add(a, b) = a + b")
    res = gs.execute("add(1, 3, 1)")
    assert not res.ok() and "too many arguments" in str(res.message)


def test_substitution_rules(gs: GraphSession):
    gs.execute(r"g(x) = \left|x\right|")

    res = gs.force_resolve_function(r"g(x)")
    assert res.ok() and res.message == "Abs(x)"

    gs.add_sub_rule(pattern="Abs", replacement="abs")
    res = gs.force_resolve_function(r"g(x)")
    assert res.ok() and res.message == "abs(x)"

    gs.execute(r"pow(x, n) = x^{n}")
    res = gs.force_resolve_function(r"pow(x, (2+5))")
    assert res.ok() and res.message == "x**(2 + 5)"

    gs.add_sub_rule(pattern=r"\*\*", replacement="^")
    res = gs.force_resolve_function(r"pow(x, (2+5))")
    assert res.ok() and res.message == "x^(2 + 5)"


def test_substitution_rule_removal(gs: GraphSession):
    gs.execute(r"pow(x, n) = x^{n}")
    res = gs.force_resolve_function(r"pow(x, (2+5))")
    assert res.ok() and res.message == "x**(2 + 5)"

    gs.add_sub_rule(pattern=r"\*\*", replacement="^")
    res = gs.force_resolve_function(r"pow(x, (2+5))")
    assert res.ok() and res.message == "x^(2 + 5)"

    gs.remove_sub_rule(pattern=r"\*\*")
    res = gs.force_resolve_function(r"pow(x, (2+5))")
    assert res.ok() and res.message == "x**(2 + 5)"


def test_substitution_rule_disabled(gs: GraphSession):
    gs.execute(r"pow(x, n) = x^{n}")
    res = gs.force_resolve_function(r"pow(x, (2+5))")
    assert res.ok() and res.message == "x**(2 + 5)"

    gs.add_sub_rule(pattern=r"\*\*", replacement="^")
    res = gs.force_resolve_function(r"pow(x, (2+5))")
    assert res.ok() and res.message == "x^(2 + 5)"

    res = gs.force_resolve_function(r"pow(x, (2+5))", use_sub_rule=False)
    assert res.ok() and res.message == "x**(2 + 5)"


def test_get_sub_rules(gs: GraphSession):
    gs.add_sub_rule(pattern=r"\*\*", replacement="^")
    gs.add_sub_rule(pattern="Abs", replacement="abs")

    rules = gs.get_sub_rules()

    assert len(rules) == 2 and rules[r"\*\*"] == "^" and rules["Abs"] == "abs"
