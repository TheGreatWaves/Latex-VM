from latexvm.graph_session import GraphSession
from latexvm.type_defs import EnvironmentVariables


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
    assert gs.get_env().get("var1") == "10"

    gs.execute("mult(n, x) = n x")
    gs.execute("var2 = mult(5, 2)")
    assert gs.get_env().get("var2") == "10"

    gs.execute("mult(n, x) = n x")
    gs.execute("var3 = mult(5, x)")
    assert gs.get_env().get("var3") == "10"


def test_func_name_absolute(gs: GraphSession):
    gs.execute("f(x) = x")
    gs.execute("some_f(x) = x x")

    gs.execute("v1 = f(2)")
    gs.execute("v2 = some_f(2)")

    env_vars: EnvironmentVariables = gs.get_env()
    assert env_vars is not None

    assert env_vars.get("v1") == "2"
    assert env_vars.get("v2") == "4"


def test_empty_execution(gs: GraphSession):
    assert len(gs.get_env()) == 0
    gs.execute("")
    assert len(gs.get_env()) == 0


def test_clear_env(gs: GraphSession):
    gs.execute("f(x) = x + x + x + x")
    gs.execute("x = 2 + 2 + 2 + 2")
    gs.execute("h = 2 + 1 + 3")

    assert len(gs.get_env()) == 3

    gs.clear_session()

    assert len(gs.get_env()) == 0
