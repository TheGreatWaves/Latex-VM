import pytest

from python_parse_test import GraphSession


@pytest.fixture()
def gs():
    gs = GraphSession.new()
    return gs
