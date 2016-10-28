import pytest

def pytest_addoption(parser):
    parser.addoption("--use-full-corpus", action="store_true", default=False)

@pytest.fixture
def use_full_corpus(request):
    return request.config.getoption("--use-full-corpus")
