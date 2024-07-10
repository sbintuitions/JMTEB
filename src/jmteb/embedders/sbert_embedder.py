from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from jmteb.embedders.base import TextEmbedder


class SentenceBertEmbedder(TextEmbedder):
    """SentenceBERT embedder."""

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        device: str | None = None,
        normalize_embeddings: bool = False,
        max_seq_length: int | None = None,
        add_eos: bool = False,
        truncate_dim: int | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
    ) -> None:
        model_kwargs = self._model_kwargs_parser(model_kwargs)
        self.model = SentenceTransformer(
            model_name_or_path,
            trust_remote_code=True,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,  # https://github.com/UKPLab/sentence-transformers/blob/84f69fee6dcde023f46a8807e89bc99a7700ba82/sentence_transformers/SentenceTransformer.py#L81-L105  # noqa: E501
            tokenizer_kwargs=tokenizer_kwargs,
        )
        if max_seq_length:
            self.model.max_seq_length = max_seq_length

        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = getattr(self.model, "max_seq_length", None)
        self.add_eos = add_eos

        if "torch_dtype" in model_kwargs:
            self.set_output_tensor()
        else:
            self.set_output_np()

    def encode(self, text: str | list[str], prefix: str | None = None) -> np.ndarray:
        if self.add_eos:
            text = self._add_eos_func(text)
        return self.model.encode(
            text,
            prompt=prefix,
            convert_to_numpy=self.convert_to_numpy,
            convert_to_tensor=self.convert_to_tensor,
            batch_size=self.batch_size,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        )

    def _add_eos_func(self, text: str | list[str]) -> str | list[str]:
        try:
            eos_token = getattr(self.model.tokenizer, "eos_token")
        except AttributeError:
            return text

        if isinstance(text, str):
            return text + eos_token
        elif isinstance(text, list):
            return [t + eos_token for t in text]

    def get_output_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
