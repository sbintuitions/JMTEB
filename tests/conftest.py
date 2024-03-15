import pytest

from src.embedders.sbert_embedder import SentenceBertEmbedder


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: pytest.Config, items: pytest.Parser):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="module")
def embedder(model_name_or_path: str = "prajjwal1/bert-tiny"):
    return SentenceBertEmbedder(model_name_or_path=model_name_or_path)
