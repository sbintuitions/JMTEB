from __future__ import annotations

import numpy as np
import pytest

from jmteb.embedders import TextEmbedder


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


class DummyTextEmbedder(TextEmbedder):

    def encode(self, text: str | list[str], prefix: str | None = None) -> np.ndarray:
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)

        return np.random.random((batch_size, self.get_output_dim()))

    def get_output_dim(self) -> int:
        return 32


@pytest.fixture(scope="module")
def embedder():
    return DummyTextEmbedder()
