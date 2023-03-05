import pytest

from graph_session import GraphSession


@pytest.fixture()
def gs():
    gs = GraphSession.new()
    return gs
