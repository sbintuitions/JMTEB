from __future__ import annotations

import numpy as np
import torch
from loguru import logger
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
        tokenizer_kwargs: dict | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name_or_path, trust_remote_code=True, tokenizer_kwargs=tokenizer_kwargs)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length

        self.batch_size = batch_size
        # When device is None and multi-GPU is available, do encoding with `model.encode` which is single-thread.
        # About setting `device` with multi-thread, refer to:
        #     https://github.com/UKPLab/sentence-transformers/blob/e5c15a51a72a5432370c4daa1d0ef7be67b4ce50/sentence_transformers/SentenceTransformer.py#L703-L706  # noqa: E501
        self.device = device
        n_available_gpus = torch.cuda.device_count()
        if n_available_gpus > 1:
            self.device = [f"cuda:{i}" for i in range(n_available_gpus)]
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = getattr(self.model, "max_seq_length", None)
        self.add_eos = add_eos

    def encode(self, text: str | list[str], prefix: str | None = None) -> np.ndarray:
        if self.add_eos:
            text = self._add_eos_func(text)
        if self.device and isinstance(text, list):
            try:
                return self._encode_multi_process(text, prefix=prefix)
            except Exception as e:
                logger.error(str(e))
                return self._encode_mono_process(text, prefix=prefix)
        else:
            return self._encode_mono_process(text, prefix=prefix)

    def _encode_mono_process(self, text: str | list[str], prefix: str | None = None) -> np.ndarray:
        return self.model.encode(
            text,
            prompt=prefix,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        )

    def _encode_multi_process(self, text: list[str], prefix: str | None = None) -> np.ndarray:
        pool = self.model.start_multi_process_pool(target_devices=self.device)
        embeddings = self.model.encode_multi_process(
            sentences=text,
            prompt=prefix,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            pool=pool,
        )
        self.model.stop_multi_process_pool(pool)
        return embeddings

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
