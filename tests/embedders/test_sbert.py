from unittest.mock import patch

import numpy as np

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

    def test_multi_process(self):
        with patch.object(SentenceBertEmbedder, "_encode_multi_process") as mock_method:
            model = SentenceBertEmbedder(MODEL_NAME_OR_PATH, device=["cpu, cpu"])
            model.encode(["任意のテキスト", "任意のテキスト"])
            mock_method.assert_called_once()

    def test_multi_process_invalid_when_text_is_str(self):
        with patch.object(SentenceBertEmbedder, "_encode_multi_process") as mock_multi_method:
            with patch.object(SentenceBertEmbedder, "_encode_mono_process") as mock_mono_method:
                model = SentenceBertEmbedder(MODEL_NAME_OR_PATH, device=["cpu, cpu"])
                model.encode("任意のテキスト")
                mock_multi_method.assert_not_called()
                mock_mono_method.assert_called_once()
