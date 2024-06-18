from __future__ import annotations

from abc import ABC
from os import PathLike
from pathlib import Path

import numpy as np
import tqdm
from loguru import logger


class TextEmbedder(ABC):
    """
    The base class of text embedder.
    """

    def encode(self, text: str | list[str], prefix: str | None = None) -> np.ndarray:
        """Convert a text string or a list of texts to embedding.

        Args:
            text (str | list[str]): text string, or a list of texts.
            prefix (str, optional): the prefix to use for encoding. Default to None.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """Get the dimensionality of output embedding

        Returns:
            int: dimensionality of output embedding
        """
        raise NotImplementedError

    def _batch_encode_and_save_on_disk(
        self,
        text_list: list[str],
        save_path: str | PathLike[str],
        prefix: str | None = None,
        batch_size: int = 64,
        dtype: str = "float32",
    ) -> np.memmap:
        """
        Encode a list of texts and save the embeddings on disk using memmap.

        Args:
            text_list (list[str]): list of texts
            save_path (str): path to save the embeddings
            prefix (str, optional): the prefix to use for encoding. Default to None.
            dtype (str, optional): data type. Defaults to "float32".
            batch_size (int): batch size. Defaults to 64.
        """

        num_samples = len(text_list)
        output_dim = self.get_output_dim()
        embeddings = np.memmap(save_path, dtype=dtype, mode="w+", shape=(num_samples, output_dim))

        with tqdm.tqdm(total=num_samples, desc="Encoding") as pbar:
            for i in range(0, num_samples, batch_size):
                batch = text_list[i : i + batch_size]
                batch_embeddings = self.encode(batch, prefix=prefix)
                batch_embeddings = np.asarray(batch_embeddings, dtype=dtype)
                embeddings[i : i + batch_size] = batch_embeddings
                pbar.update(len(batch))
        embeddings.flush()

        return np.memmap(save_path, dtype=dtype, mode="r", shape=(num_samples, output_dim))

    def batch_encode_with_cache(
        self,
        text_list: list[str],
        prefix: str | None = None,
        cache_path: str | PathLike[str] | None = None,
        overwrite_cache: bool = False,
        batch_size: int = 64,
        dtype: str = "float32",
    ) -> np.ndarray:
        """
        Encode a list of texts and save the embeddings on disk using memmap if cache_path is provided.

        Args:
            text_list (list[str]): list of texts
            prefix (str, optional): the prefix to use for encoding. Default to None.
            cache_path (str, optional): path to save the embeddings. Defaults to None.
            overwrite_cache (bool, optional): whether to overwrite the cache. Defaults to False.
            batch_size (int): batch size. Defaults to 64.
            dtype (str, optional): data type. Defaults to "float32".
        """

        if cache_path is None:
            logger.info("Encoding embeddings")
            return self.encode(text_list, prefix=prefix).astype(dtype)

        if Path(cache_path).exists() and not overwrite_cache:
            logger.info(f"Loading embeddings from {cache_path}")
            return np.memmap(cache_path, dtype=dtype, mode="r", shape=(len(text_list), self.get_output_dim()))

        logger.info(f"Encoding and saving embeddings to {cache_path}")
        embeddings = self._batch_encode_and_save_on_disk(
            text_list, cache_path, prefix=prefix, batch_size=batch_size, dtype=dtype
        )
        return embeddings
