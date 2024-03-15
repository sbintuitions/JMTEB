import pytest

from src.embedders.sbert_embedder import SentenceBertEmbedder


@pytest.fixture(scope="module")
def embedder(model_name_or_path: str = "prajjwal1/bert-tiny"):
    return SentenceBertEmbedder(model_name_or_path=model_name_or_path)
