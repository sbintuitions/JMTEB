from __future__ import annotations

from abc import ABC
from os import PathLike
from pathlib import Path

import numpy as np
import torch
import tqdm
from loguru import logger


class TextEmbedder(ABC):
    """
    The base class of text embedder.
    """

    convert_to_tensor: bool
    convert_to_numpy: bool
    _chunk_size: int = 262144  # 2^18

    def encode(self, text: str | list[str], prefix: str | None = None) -> np.ndarray | torch.Tensor:
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
        batch_size: int = 262144,
        dtype: str = "float32",
    ) -> np.memmap | torch.Tensor:
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
        if self.convert_to_numpy:
            embeddings = np.memmap(save_path, dtype=dtype, mode="w+", shape=(num_samples, output_dim))
        else:
            embeddings = torch.empty((num_samples, output_dim), dtype=self._torch_dtype_parser(dtype))

        with tqdm.tqdm(total=num_samples, desc="Encoding") as pbar:
            for i in range(0, num_samples, batch_size):
                batch = text_list[i : i + batch_size]
                batch_embeddings: np.ndarray | torch.Tensor = self.encode(batch, prefix=prefix)
                embeddings[i : i + batch_size] = batch_embeddings
                pbar.update(len(batch))

        if self.convert_to_numpy:
            embeddings.flush()
            return np.memmap(save_path, dtype=dtype, mode="r", shape=(num_samples, output_dim))
        else:
            torch.save(embeddings, save_path)
            return embeddings

    def batch_encode_with_cache(
        self,
        text_list: list[str],
        prefix: str | None = None,
        cache_path: str | PathLike[str] | None = None,
        overwrite_cache: bool = False,
        dtype: str = "float32",
    ) -> np.ndarray | torch.Tensor:
        """
        Encode a list of texts and save the embeddings on disk using memmap if cache_path is provided.

        Args:
            text_list (list[str]): list of texts
            prefix (str, optional): the prefix to use for encoding. Default to None.
            cache_path (str, optional): path to save the embeddings. Defaults to None.
            overwrite_cache (bool, optional): whether to overwrite the cache. Defaults to False.
            dtype (str, optional): data type. Defaults to "float32".
        """

        if cache_path is None:
            logger.info("Encoding embeddings")
            return self.encode(text_list, prefix=prefix)

        if Path(cache_path).exists() and not overwrite_cache:
            logger.info(f"Loading embeddings from {cache_path}")
            return np.memmap(cache_path, dtype=dtype, mode="r", shape=(len(text_list), self.get_output_dim()))

        logger.info(f"Encoding and saving embeddings to {cache_path}")
        embeddings = self._batch_encode_and_save_on_disk(
            text_list, cache_path, prefix=prefix, batch_size=self._chunk_size, dtype=dtype
        )
        return embeddings

    @staticmethod
    def _torch_dtype_parser(dtype: str | torch.dtype) -> torch.dtype | str:
        if dtype == "auto":
            return dtype
        elif isinstance(dtype, str):
            dtype = dtype.replace("torch.", "")
            if hasattr(torch, dtype):
                dtype = getattr(torch, dtype)
                if isinstance(dtype, torch.dtype):
                    return dtype
            raise ValueError(f"Invalid torch dtype: {dtype}")
        elif isinstance(dtype, torch.dtype):
            return dtype
        else:
            raise ValueError(f"Expected `dtype` as `str` or `torch.dtype`, but got {type(dtype)}!")

    def _model_kwargs_parser(self, model_kwargs: dict | None) -> dict:
        if not model_kwargs:
            return {}

        if "torch_dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = self._torch_dtype_parser(model_kwargs["torch_dtype"])
        return model_kwargs

    def set_output_tensor(self):
        self.convert_to_numpy = False
        self.convert_to_tensor = True

    def set_output_numpy(self):
        self.convert_to_numpy = True
        self.convert_to_tensor = False
