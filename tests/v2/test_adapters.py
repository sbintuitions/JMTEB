"""
Tests for JMTEB v2.0 adapters (JMTEBModel).
"""

from unittest.mock import Mock, patch

import mteb
import numpy as np
import pytest
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import PromptType
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from jmteb.embedders.base import TextEmbedder
from jmteb.v2.adapters import JMTEBModel, TextEncoderWrapper


class CustomModel:
    """Custom model that only encodes a plain list of texts."""

    def __init__(self):
        self.calls = []

    def encode(self, sentences, batch_size=32, **kwargs):
        self.calls.append({"sentences": sentences, "batch_size": batch_size})
        return np.ones((len(sentences), 4))


class FakeSentenceTransformer:
    """Stands in for the SentenceTransformer class to record constructor arguments."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.prompts = kwargs.get("prompts") or {}


class DummyEmbedder(TextEmbedder):
    """JMTEB v1 TextEmbedder stub."""

    def __init__(self):
        self.calls = []

    def encode(self, text, prefix=None, **kwargs):
        self.calls.append({"text": text, "prefix": prefix})
        return np.ones((len(text), 4))


def make_dataloader(sentences, batch_size=2):
    """DataLoader yielding batches in MTEB's format ({"text": [...]})."""
    return DataLoader([{"text": s} for s in sentences], batch_size=batch_size)


def encode(model, sentences, task_name="JSTS", prompt_type=None, **kwargs):
    return model.encode(
        make_dataloader(sentences),
        task_metadata=mteb.get_task(task_name).metadata,
        hf_split="test",
        hf_subset="default",
        prompt_type=prompt_type,
        **kwargs,
    )


class TestJMTEBModel:
    """Tests for JMTEBModel adapter."""

    def test_init_with_embedder(self, mock_embedder):
        """Test initialization with v1 embedder."""
        model = JMTEBModel(embedder=mock_embedder)
        assert model.embedder == mock_embedder
        assert model.sentence_transformer is None

    def test_init_with_sentence_transformer(self, mock_sentence_transformer):
        """Test initialization with SentenceTransformer."""
        model = JMTEBModel(sentence_transformer=mock_sentence_transformer)
        assert model.sentence_transformer == mock_sentence_transformer
        assert model.embedder is None

    def test_init_without_model_raises_error(self):
        """Test that initialization without model raises error."""
        with pytest.raises(ValueError, match="Either embedder or sentence_transformer must be provided"):
            JMTEBModel()

    def test_implements_mteb_encoder_protocol(self):
        """MTEB only evaluates models that implement its encoder protocol."""
        model = JMTEBModel(sentence_transformer=CustomModel(), model_name="custom")
        assert isinstance(model, EncoderProtocol)
        assert model.mteb_model_meta.name == "custom"

    def test_encode_with_custom_model(self, sample_sentences):
        """Test encoding a DataLoader with a model that only encodes plain texts."""
        custom_model = CustomModel()
        model = JMTEBModel(sentence_transformer=custom_model)
        result = encode(model, sample_sentences, batch_size=16)

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(sample_sentences), 4)
        assert custom_model.calls == [{"sentences": sample_sentences, "batch_size": 16}]

    def test_encode_with_custom_model_and_prompts(self, sample_sentences):
        """Test that prompts are selected by task type and prompt type and prepended."""
        custom_model = CustomModel()
        prompts = {"Retrieval-query": "query: ", "Retrieval-document": "passage: "}
        model = JMTEBModel(sentence_transformer=custom_model, prompts=prompts)
        encode(model, sample_sentences, task_name="JaqketRetrieval", prompt_type=PromptType.query)

        assert custom_model.calls[0]["sentences"] == ["query: " + s for s in sample_sentences]

    def test_encode_with_embedder(self, sample_sentences):
        """Test encoding with v1 embedder, passing the prompt as prefix."""
        embedder = DummyEmbedder()
        model = JMTEBModel.from_jmteb_embedder(embedder, prompts={"STS": "sts: "}, model_name="dummy")
        result = encode(model, sample_sentences)

        assert result.shape == (len(sample_sentences), 4)
        assert embedder.calls == [{"text": sample_sentences, "prefix": "sts: "}]
        assert model.mteb_model_meta.name == "dummy"

    @patch("jmteb.v2.adapters.SentenceTransformerEncoderWrapper")
    def test_encode_with_sentence_transformer(self, mock_wrapper_class, sample_sentences):
        """Test that SentenceTransformer models are encoded via MTEB's wrapper with merged kwargs."""
        st_model = Mock(spec=SentenceTransformer)
        st_model.prompts = {"query": ""}
        model = JMTEBModel(sentence_transformer=st_model, prompts={"STS": "sts: "}, show_progress_bar=False)

        mock_wrapper_class.assert_called_once_with(st_model)
        assert st_model.prompts == {"query": "", "STS": "sts: "}

        encode(model, sample_sentences, batch_size=64)
        call_kwargs = mock_wrapper_class.return_value.encode.call_args[1]
        assert call_kwargs["batch_size"] == 64
        assert call_kwargs["show_progress_bar"] is False
        assert call_kwargs["task_metadata"].name == "JSTS"

    def test_mteb_model_used_as_is(self, sample_sentences):
        """Test that models loaded with mteb.get_model are used directly."""
        mteb_model = Mock()
        model = JMTEBModel(sentence_transformer=mteb_model)

        assert model.mteb_model_meta is mteb_model.mteb_model_meta
        encode(model, sample_sentences)
        mteb_model.encode.assert_called_once()

    @patch("jmteb.v2.adapters.SentenceTransformerEncoderWrapper")
    @patch("jmteb.v2.adapters.SentenceTransformer", FakeSentenceTransformer)
    def test_from_sentence_transformer(self, mock_wrapper_class):
        """Test creating model from sentence transformer path."""
        model = JMTEBModel.from_sentence_transformer(
            "test-model",
            device="cuda",
            model_kwargs={"torch_dtype": "float16"},
        )

        # Check SentenceTransformer was called correctly
        st_model = model.sentence_transformer
        assert isinstance(st_model, FakeSentenceTransformer)
        assert st_model.init_kwargs["model_name_or_path"] == "test-model"
        assert st_model.init_kwargs["device"] == "cuda"

        # Check model was wrapped
        mock_wrapper_class.assert_called_once_with(st_model)

    def test_from_jmteb_embedder(self, mock_embedder):
        """Test creating model from v1 embedder."""
        prompts = {"query": "query: ", "document": "passage: "}
        model = JMTEBModel.from_jmteb_embedder(mock_embedder, prompts=prompts)

        assert model.embedder == mock_embedder
        assert model.prompts == prompts


class TestTextEncoderWrapper:
    """Tests for TextEncoderWrapper."""

    def test_invalid_prompt_keys_are_ignored(self):
        wrapper = TextEncoderWrapper(CustomModel(), model_name="custom", prompts={"STS": "s: ", "bogus": "x"})
        assert wrapper.model_prompts == {"STS": "s: "}
