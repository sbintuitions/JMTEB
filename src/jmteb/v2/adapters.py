"""
Adapters to bridge models with the MTEB (>= 2.x) evaluation framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from loguru import logger
from mteb.models import ModelMeta, SentenceTransformerEncoderWrapper
from mteb.models.abs_encoder import AbsEncoder
from sentence_transformers import SentenceTransformer

from jmteb.embedders.base import TextEmbedder

if TYPE_CHECKING:
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from torch.utils.data import DataLoader


class TextEncoderWrapper(AbsEncoder):
    """
    MTEB encoder for models that only encode a plain list of texts.

    Wraps a JMTEB v1 TextEmbedder (called as ``encode(texts, prefix=prompt)``) or any object with
    ``encode(sentences: list[str], batch_size: int) -> array`` (prompts are prepended to the texts).
    """

    def __init__(self, model: Any, model_name: str, prompts: dict[str, str] | None = None):
        self.model = model
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={"name": model_name})
        if prompts:
            self.model_prompts, invalid_prompts = self.validate_task_to_prompt_name(
                prompts, raise_for_invalid_keys=False
            )
            if invalid_prompts:
                logger.warning(f"Some prompts are not in the expected format and will be ignored: {invalid_prompts}")

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> Array:
        sentences = [text for batch in inputs for text in batch["text"]]
        prompt_name = self.get_prompt_name(task_metadata, prompt_type)
        prompt = self.model_prompts.get(prompt_name) if prompt_name else None

        if isinstance(self.model, TextEmbedder):
            embeddings = self.model.encode(sentences, prefix=prompt)
        else:
            if prompt:
                sentences = [prompt + sentence for sentence in sentences]
            embeddings = self.model.encode(sentences, batch_size=kwargs.get("batch_size", 32))

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().float().numpy()
        return np.asarray(embeddings)


class JMTEBModel:
    """
    Adapter that makes a model usable with MTEB's evaluation system.

    It implements MTEB's encoder interface (``encode`` on a DataLoader, ``similarity``,
    ``mteb_model_meta``) and delegates to an encoder built from the wrapped model:

    - SentenceTransformer: MTEB's SentenceTransformerEncoderWrapper (prompts are resolved by MTEB)
    - Models from ``mteb.get_model``: used as-is
    - JMTEB v1 TextEmbedder, or any object with ``encode(list[str], batch_size)``: TextEncoderWrapper

    Example:
        >>> from jmteb.v2.adapters import JMTEBModel
        >>>
        >>> model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-v3-30m")
        >>>
        >>> # Now use with MTEB
        >>> import mteb
        >>> tasks = mteb.get_tasks(tasks=["JSTS"], languages=["jpn"])
        >>> results = mteb.evaluate(model, tasks=tasks)
    """

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        sentence_transformer: Any | None = None,
        prompts: dict[str, str] | None = None,
        model_name: str | None = None,
        **encode_kwargs,
    ):
        """
        Initialize the JMTEB model adapter.

        Args:
            embedder: JMTEB v1 TextEmbedder instance (for backward compatibility)
            sentence_transformer: SentenceTransformer model, a model from ``mteb.get_model``,
                or any object with ``encode(sentences: list[str], batch_size: int)``
            prompts: Dictionary mapping task names/types (optionally suffixed with ``-query`` or
                ``-document``) to prompts
            model_name: Name used by MTEB to cache results. Only used for v1 embedders and custom
                models (defaults to the class name); otherwise the name comes from the model itself.
            **encode_kwargs: Additional keyword arguments passed to the SentenceTransformer or MTEB
                model's encode method
        """
        if embedder is None and sentence_transformer is None:
            raise ValueError("Either embedder or sentence_transformer must be provided")

        self.embedder = embedder
        self.sentence_transformer = sentence_transformer
        self.prompts = prompts or {}
        self.encode_kwargs = encode_kwargs
        self._encoder = self._build_encoder(model_name)

    def _build_encoder(self, model_name: str | None) -> Any:
        model = self.embedder if self.embedder is not None else self.sentence_transformer

        if isinstance(model, SentenceTransformer):
            if self.prompts:
                # MTEB's wrapper selects prompts from model.prompts
                model.prompts = {**(model.prompts or {}), **self.prompts}
            return SentenceTransformerEncoderWrapper(model)

        if hasattr(model, "mteb_model_meta"):
            if self.prompts:
                logger.warning("prompts are ignored for models loaded with MTEB; the model's own prompts are used")
            return model

        if model_name is None:
            model_name = type(model).__name__
            logger.warning(
                f"No model_name given for {model_name}; results will be cached under '{model_name}'. "
                "Pass model_name to JMTEBModel to keep results of different models apart."
            )
        return TextEncoderWrapper(model, model_name=model_name, prompts=self.prompts)

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._encoder.mteb_model_meta

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> Array:
        """
        Encode inputs following MTEB's encoder interface.

        Args:
            inputs: DataLoader of batched inputs provided by MTEB
            task_metadata: Metadata of the task being evaluated (used to select prompts)
            hf_split: Split of the current task
            hf_subset: Subset of the current task
            prompt_type: Prompt type (query or document)
            **kwargs: Additional encoding arguments (e.g. batch_size)

        Returns:
            Array of embeddings with shape (num_inputs, embedding_dim)
        """
        return self._encoder.encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **{**self.encode_kwargs, **kwargs},
        )

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        return self._encoder.similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        return self._encoder.similarity_pairwise(embeddings1, embeddings2)

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
            prompts: Dictionary mapping task names/types to prompts
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
        model_name: str | None = None,
        **encode_kwargs,
    ) -> JMTEBModel:
        """
        Create a JMTEBModel from a JMTEB v1 TextEmbedder.

        Args:
            embedder: JMTEB v1 TextEmbedder instance
            prompts: Dictionary mapping task names/types to prompts (passed to the embedder as prefix)
            model_name: Name used by MTEB to cache results (defaults to the embedder's class name)
            **encode_kwargs: Additional encoding keyword arguments

        Returns:
            JMTEBModel instance
        """
        return cls(
            embedder=embedder,
            prompts=prompts,
            model_name=model_name,
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
