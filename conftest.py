import pytest

def pytest_addoption(parser):
    parser.addoption('--corpus_percentage', type=float, default=1.0, help='e.g. 1 == full corpus; 0.5 == 50% of corpus')


@pytest.fixture
def corpus_percentage(request):
    return request.config.getoption("--corpus_percentage")


def pytest_addoption(parser):
    parser.addoption('--evaluation_level', type=int)


@pytest.fixture
def evaluation_level(request):
    return request.config.getoption("--evaluation_level")
