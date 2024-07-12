import numpy as np
import torch

from jmteb.embedders.transformers_embedder import TransformersEmbedder

MODEL_NAME_OR_PATH = "prajjwal1/bert-tiny"
OUTPUT_DIM = 128


class TestTransformersEmbedder:
    def setup_class(cls):
        cls.model = TransformersEmbedder(MODEL_NAME_OR_PATH)

    def test_encode(self):
        embeddings = self.model.encode("任意のテキスト")
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (OUTPUT_DIM,)

    def test_encode_list(self):
        embeddings = self.model.encode(["任意のテキスト", "hello world", "埋め込み"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, OUTPUT_DIM)

    def test_get_output_dim(self):
        assert self.model.get_output_dim() == OUTPUT_DIM

    def test_tokenizer_kwargs(self):
        assert self.model.tokenizer.sep_token == "[SEP]"
        model = TransformersEmbedder(MODEL_NAME_OR_PATH, tokenizer_kwargs={"sep_token": "<sep>"})
        assert model.tokenizer.sep_token == "<sep>"

    def test_model_kwargs(self):
        model = TransformersEmbedder(MODEL_NAME_OR_PATH, model_kwargs={"torch_dtype": torch.float16})
        assert model.convert_to_tensor
        assert model.encode("任意のテキスト").dtype is torch.float16

    def test_bf16(self):
        # As numpy doesn't support native bfloat16, add a test case for bf16
        model = TransformersEmbedder(MODEL_NAME_OR_PATH, model_kwargs={"torch_dtype": torch.bfloat16})
        assert model.convert_to_tensor
        assert model.encode("任意のテキスト").dtype is torch.bfloat16
