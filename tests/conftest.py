import pytest

from latexvm.graph_session import GraphSession


@pytest.fixture()
def gs():
    gs = GraphSession.new()
    yield gs
    gs.clear_session()
