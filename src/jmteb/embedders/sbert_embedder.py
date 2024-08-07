from __future__ import annotations

from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer

from jmteb.embedders.base import TextEmbedder


@contextmanager
def sbert_multi_proc_pool(sbert_model: SentenceTransformer, target_devices: Optional[list[str]] = None):
    pool = sbert_model.start_multi_process_pool(target_devices=target_devices)
    logger.info("pool of encoding processing: ")
    for k, v in pool.items():
        logger.info(f"  {k}: {v}")
    try:
        yield pool
    finally:
        logger.info("stop pool")
        sbert_model.stop_multi_process_pool(pool)


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
        chunk_size_factor: int = 128,
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
        self.set_output_numpy()
        self.model.eval()
        self.chunk_size_factor = chunk_size_factor

    # override
    def _batch_encode_and_save_on_disk(
        self,
        text_list: list[str],
        save_path: str | PathLike[str],
        prefix: str | None = None,
        batch_size: int = 262144,
        dtype: str = "float32",
    ) -> np.memmap:
        """
        Encode a list of texts and save the embeddings on disk using memmap.

        Args:
            text_list (list[str]): list of texts
            save_path (str): path to save the embeddings
            prefix (str, optional): the prefix to use for encoding. Default to None.
            dtype (str, optional): data type. Defaults to "float32".
            batch_size (int): batch size. Defaults to 262144.
        """
        self.set_output_numpy()
        self.model.eval()
        logger.info(f"use numpy")

        num_samples = len(text_list)
        output_dim = self.get_output_dim()

        embeddings = np.memmap(save_path, dtype=dtype, mode="w+", shape=(num_samples, output_dim))

        with sbert_multi_proc_pool(self.model) as pool:
            with tqdm.tqdm(total=num_samples, desc="Encoding") as pbar:
                chunk_size = int(min(
                    self.batch_size * self.chunk_size_factor,
                    np.ceil(num_samples / len(pool["processes"])),
                ))
                logger.info(f"chunk size={chunk_size}")
                for i in range(0, num_samples, batch_size):
                    batch: list[str] = text_list[i : i + batch_size]
                    batch = self._add_eos_func(batch)
                    batch_embeddings: np.ndarray = self.model.encode_multi_process(
                        batch,
                        pool=pool,
                        prompt=prefix,
                        chunk_size=chunk_size,
                        batch_size=self.batch_size,
                        normalize_embeddings=self.normalize_embeddings,
                    )
                    embeddings[i : i + batch_size] = batch_embeddings
                    pbar.update(len(batch))

        embeddings.flush()
        return np.memmap(save_path, dtype=dtype, mode="r", shape=(num_samples, output_dim))

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
