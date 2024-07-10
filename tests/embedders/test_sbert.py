import numpy as np
import torch

from jmteb.embedders.sbert_embedder import SentenceBertEmbedder

MODEL_NAME_OR_PATH = "prajjwal1/bert-tiny"
OUTPUT_DIM = 128


class TestSentenceBertEmbedder:
    def setup_class(cls):
        cls.model = SentenceBertEmbedder(MODEL_NAME_OR_PATH)

    def test_encode(self):
        embeddings = self.model.encode("任意のテキスト")
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (OUTPUT_DIM,)

    def test_get_output_dim(self):
        assert self.model.get_output_dim() == OUTPUT_DIM

    def test_tokenizer_kwargs(self):
        assert self.model.model.tokenizer.sep_token == "[SEP]"
        model = SentenceBertEmbedder(MODEL_NAME_OR_PATH, tokenizer_kwargs={"sep_token": "<sep>"})
        assert model.model.tokenizer.sep_token == "<sep>"

    def test_model_kwargs(self):
        model = SentenceBertEmbedder(MODEL_NAME_OR_PATH, model_kwargs={"torch_dtype": torch.float16})
        assert model.convert_to_tensor
        assert model.encode("任意のテキスト").dtype is torch.float16
