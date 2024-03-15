from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedders.base import TextEmbedder


class SentenceBertEmbedder(TextEmbedder):
    """SentenceBERT embedder."""

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        device: str | None = None,
        normalize_embeddings: bool = False,
    ) -> None:
        self.model = SentenceTransformer(model_name_or_path)
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings

    def encode(self, text: str | list[str]) -> np.ndarray:
        return self.model.encode(
            text,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        )

    def get_output_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
