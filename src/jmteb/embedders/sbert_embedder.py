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
        tokenizer_padding_side: str | None = None,
        add_eos: bool = False,
    ) -> None:
        self.model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        if tokenizer_padding_side:
            try:
                self.model.tokenizer.padding_side = "right"
            except AttributeError:
                pass

        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.tokenizer_padding_side = tokenizer_padding_side
        self.add_eos = add_eos

        if self.max_seq_length:
            self.model.max_seq_length = self.max_seq_length
        if self.tokenizer_padding_side:
            setattr(self.model.tokenizer, "padding_side", self.tokenizer_padding_side)

    def encode(self, text: str | list[str], prompt: str | None = None) -> np.ndarray:
        if self.add_eos:
            text = self.add_eos_func(text)
        return self.model.encode(
            text,
            prompt=prompt,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        )

    def add_eos_func(self, text: str | list[str]) -> str | list[str]:
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
