import numpy as np

from jmteb.embedders.openai_embedder import OpenAIEmbedder

OUTPUT_DIM = 1536


class TestSentenceBertEmbedder:
    def setup_class(cls):
        cls.model = OpenAIEmbedder(model="text-embedding-3-small")

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
