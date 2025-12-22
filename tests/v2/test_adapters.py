"""
Tests for JMTEB v2.0 adapters (JMTEBModel).
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from jmteb.v2.adapters import JMTEBModel


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
        with pytest.raises(
            ValueError, match="Either embedder or sentence_transformer must be provided"
        ):
            JMTEBModel()

    def test_encode_with_embedder(self, mock_embedder, sample_sentences):
        """Test encoding with v1 embedder."""
        model = JMTEBModel(embedder=mock_embedder)
        result = model.encode(sample_sentences, batch_size=32)

        # Check that embedder.encode was called
        mock_embedder.encode.assert_called_once()

        # Check result is numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 10  # Mock returns 10 embeddings

    def test_encode_with_sentence_transformer(
        self, mock_sentence_transformer, sample_sentences
    ):
        """Test encoding with SentenceTransformer."""
        model = JMTEBModel(sentence_transformer=mock_sentence_transformer)
        result = model.encode(sample_sentences, batch_size=32)

        # Check that st.encode was called
        mock_sentence_transformer.encode.assert_called_once()

        # Check result is numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 10

    def test_encode_with_kwargs(self, mock_sentence_transformer, sample_sentences):
        """Test encoding with additional kwargs."""
        model = JMTEBModel(
            sentence_transformer=mock_sentence_transformer, show_progress_bar=False
        )
        result = model.encode(sample_sentences, batch_size=64, prompt_name="query")

        # Verify kwargs were passed
        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["batch_size"] == 64
        assert call_kwargs["show_progress_bar"] is False

        # Check result
        assert isinstance(result, np.ndarray)

    @patch("jmteb.v2.adapters.SentenceTransformer")
    def test_from_sentence_transformer(self, mock_st_class):
        """Test creating model from sentence transformer path."""
        mock_model = Mock()
        mock_st_class.return_value = mock_model

        model = JMTEBModel.from_sentence_transformer(
            "test-model",
            device="cuda",
            model_kwargs={"torch_dtype": "float16"},
        )

        # Check SentenceTransformer was called correctly
        mock_st_class.assert_called_once()
        call_kwargs = mock_st_class.call_args[1]
        assert call_kwargs["model_name_or_path"] == "test-model"
        assert call_kwargs["device"] == "cuda"

        # Check model was wrapped
        assert model.sentence_transformer == mock_model

    def test_from_jmteb_embedder(self, mock_embedder):
        """Test creating model from v1 embedder."""
        prompts = {"query": "query: ", "passage": "passage: "}
        model = JMTEBModel.from_jmteb_embedder(mock_embedder, prompts=prompts)

        assert model.embedder == mock_embedder
        assert model.prompts == prompts
