"""
Adapters to bridge JMTEB v1 embedders with MTEB evaluation framework.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from jmteb.embedders.base import TextEmbedder


class JMTEBModel:
    """
    Adapter that wraps JMTEB v1 TextEmbedder to work with MTEB's evaluation system.

    This adapter allows using existing JMTEB embedders (SentenceBertEmbedder,
    OpenAIEmbedder, etc.) with the MTEB evaluation framework while maintaining
    their specific behaviors and configurations.

    Example:
        >>> from jmteb.embedders import SentenceBertEmbedder
        >>> from jmteb.v2.adapters import JMTEBModel
        >>>
        >>> embedder = SentenceBertEmbedder(model_name_or_path="cl-nagoya/ruri-base")
        >>> model = JMTEBModel(embedder)
        >>>
        >>> # Now use with MTEB
        >>> import mteb
        >>> tasks = mteb.get_tasks(languages=["jpn"])
        >>> results = mteb.evaluate(model, tasks=tasks[:1])
    """

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        sentence_transformer: SentenceTransformer | None = None,
        prompts: dict[str, str] | None = None,
        **encode_kwargs,
    ):
        """
        Initialize the JMTEB model adapter.

        Args:
            embedder: JMTEB v1 TextEmbedder instance (for backward compatibility)
            sentence_transformer: SentenceTransformer model (for direct MTEB usage)
            prompts: Dictionary mapping task types to prompt templates
            **encode_kwargs: Additional keyword arguments for encoding
        """
        if embedder is None and sentence_transformer is None:
            raise ValueError("Either embedder or sentence_transformer must be provided")

        self.embedder = embedder
        self.sentence_transformer = sentence_transformer
        self.prompts = prompts or {}
        self.encode_kwargs = encode_kwargs

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode sentences into embeddings.

        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
            **kwargs: Additional encoding arguments (prompt_name, task_type, etc.)

        Returns:
            Array of embeddings with shape (len(sentences), embedding_dim)
        """
        # Merge default encode_kwargs with method-specific kwargs
        encode_params = {**self.encode_kwargs, **kwargs}
        encode_params["batch_size"] = batch_size

        # Use JMTEB v1 embedder if provided
        if self.embedder is not None:
            embeddings = self.embedder.encode(
                sentences,
                batch_size=batch_size,
            )
            return np.array(embeddings)

        # Otherwise use SentenceTransformer directly
        embeddings = self.sentence_transformer.encode(
            sentences,
            **encode_params,
        )
        return np.array(embeddings)

    @classmethod
    def from_sentence_transformer(
        cls,
        model_name_or_path: str,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        prompts: dict[str, str] | None = None,
        **encode_kwargs,
    ) -> JMTEBModel:
        """
        Create a JMTEBModel from a SentenceTransformer model name or path.

        Args:
            model_name_or_path: Model name on HuggingFace Hub or local path
            device: Device to run the model on
            model_kwargs: Additional keyword arguments for model initialization
            prompts: Dictionary mapping task types to prompt templates
            **encode_kwargs: Additional encoding keyword arguments

        Returns:
            JMTEBModel instance
        """
        model = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            device=device,
            model_kwargs=model_kwargs,
            prompts=prompts,
            trust_remote_code=True,
        )
        return cls(
            sentence_transformer=model,
            prompts=prompts,
            **encode_kwargs,
        )

    @classmethod
    def from_jmteb_embedder(
        cls,
        embedder: TextEmbedder,
        prompts: dict[str, str] | None = None,
        **encode_kwargs,
    ) -> JMTEBModel:
        """
        Create a JMTEBModel from a JMTEB v1 TextEmbedder.

        Args:
            embedder: JMTEB v1 TextEmbedder instance
            prompts: Dictionary mapping task types to prompt templates
            **encode_kwargs: Additional encoding keyword arguments

        Returns:
            JMTEBModel instance
        """
        return cls(
            embedder=embedder,
            prompts=prompts,
            **encode_kwargs,
        )

    @classmethod
    def from_mteb(
        cls,
        model_name: str,
        **model_kwargs,
    ) -> JMTEBModel:
        """
        Create a JMTEBModel using MTEB's get_model function.

        This method uses MTEB's unified model loading interface, which supports
        various model types and handles model-specific configurations automatically.

        Args:
            model_name: Name of the model (e.g., "sentence-transformers/all-MiniLM-L6-v2")
            **model_kwargs: Additional keyword arguments passed to mteb.get_model

        Returns:
            JMTEBModel instance

        Example:
            >>> model = JMTEBModel.from_mteb("sentence-transformers/all-MiniLM-L6-v2")
            >>> # Or with specific revision
            >>> model = JMTEBModel.from_mteb("intfloat/multilingual-e5-base", revision="main")
        """
        import mteb

        mteb_model = mteb.get_model(model_name, **model_kwargs)
        return cls(sentence_transformer=mteb_model)
