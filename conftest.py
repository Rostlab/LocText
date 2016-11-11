import pytest

def pytest_addoption(parser):
    parser.addoption('--corpus_percentage', type=float, default=0.4, help='e.g. 1 == full corpus; 0.5 == 50% of corpus')


@pytest.fixture
def corpus_percentage(request):
    return request.config.getoption("--corpus_percentage")
