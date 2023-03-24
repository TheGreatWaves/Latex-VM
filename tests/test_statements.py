from latexvm.graph_session import GraphSession


def test_statement(gs: GraphSession):
    gs.execute(r"f(x) = x^{2}")
    res = gs.execute(r"f(f(2))")

    assert res.ok() and "16" in res.message

    res1 = gs.execute(r"2")
    res2 = gs.execute(r"2.3")
    assert res1.ok() and "2.0" == res1.message
    assert res2.ok() and "2.3" == res2.message


def test_statement_fail(gs: GraphSession):
    res = gs.execute(r"y")
    assert "_lambdifygenerated()" in str(res.message)
