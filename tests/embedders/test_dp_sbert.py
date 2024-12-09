import numpy as np
import torch

from jmteb.embedders.data_parallel_sbert_embedder import (
    DataParallelSentenceBertEmbedder,
)

MODEL_NAME_OR_PATH = "prajjwal1/bert-tiny"
OUTPUT_DIM = 128


class TestDPSentenceBertEmbedder:
    def setup_class(cls):
        cls.model = DataParallelSentenceBertEmbedder(MODEL_NAME_OR_PATH)

    def test_encode(self):
        embeddings = self.model.encode("任意のテキスト")
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (OUTPUT_DIM,)

    def test_encode_with_prompt(self):
        embeddings = self.model.encode("任意のテキスト", prefix="プロンプト")
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (OUTPUT_DIM,)

    def test_get_output_dim(self):
        assert self.model.get_output_dim() == OUTPUT_DIM

    def test_tokenizer_kwargs(self):
        assert self.model.model.tokenizer.sep_token == "[SEP]"
        model = DataParallelSentenceBertEmbedder(MODEL_NAME_OR_PATH, tokenizer_kwargs={"sep_token": "<sep>"})
        assert model.model.tokenizer.sep_token == "<sep>"

    def test_model_kwargs(self):
        model = DataParallelSentenceBertEmbedder(MODEL_NAME_OR_PATH, model_kwargs={"torch_dtype": torch.float16})
        assert model.convert_to_tensor
        assert model.encode("任意のテキスト").dtype is torch.float16

    def test_bf16(self):
        # As numpy doesn't support native bfloat16, add a test case for bf16
        model = DataParallelSentenceBertEmbedder(MODEL_NAME_OR_PATH, model_kwargs={"torch_dtype": torch.bfloat16})
        assert model.convert_to_tensor
        assert model.encode("任意のテキスト").dtype is torch.bfloat16
