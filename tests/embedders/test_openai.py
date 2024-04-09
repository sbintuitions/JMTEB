from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from pytest_mock import MockerFixture

from jmteb.embedders import OpenAIEmbedder, TextEmbedder

OUTPUT_DIM = 1536


@pytest.fixture(scope="function")
def mock_openai_embedder(mocker: MockerFixture):
    mocker.patch("jmteb.embedders.openai_embedder.OpenAI")
    return OpenAIEmbedder(model="text-embedding-3-small")


@dataclass
class MockData:
    data: list


@dataclass
class MockEmbedding:
    embedding: list


class MockEmbedder:
    def create(input: str | list[str], model: str, dimensions: int):
        if isinstance(input, str):
            input = [input]
        return MockData([MockEmbedding(embedding=[0.1] * dimensions)] * len(input))


@pytest.mark.usefixtures("mock_openai_embedder")
class TestOpenAIEmbedder:
    @pytest.fixture(autouse=True)
    def setup_class(cls, mocker: MockerFixture, mock_openai_embedder: TextEmbedder):
        cls.model = mock_openai_embedder
        cls.mock_create = mocker.patch.object(cls.model.client, "embeddings", new=MockEmbedder)

    def test_encode(self):
        embeddings = self.model.encode("任意のテキスト")
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (OUTPUT_DIM,)

    def test_encode_multiple(self):
        embeddings = self.model.encode(["任意のテキスト"] * 3)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, OUTPUT_DIM)

    def test_get_output_dim(self):
        assert self.model.get_output_dim() == OUTPUT_DIM
