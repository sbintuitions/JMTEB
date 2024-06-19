from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tiktoken
from loguru import logger
from openai import OpenAI

from jmteb.embedders.base import TextEmbedder


@dataclass
class OpenAIEmbedderConfig:
    max_output_dim: int
    encoder_name: str
    max_token_length: int


OPENAI_EMBEDDERS = {
    # https://platform.openai.com/docs/guides/embeddings/embedding-models
    "text-embedding-3-large": OpenAIEmbedderConfig(3072, "cl100k_base", 8191),
    "text-embedding-3-small": OpenAIEmbedderConfig(1536, "cl100k_base", 8191),
    "text-embedding-ada-002": OpenAIEmbedderConfig(1536, "cl100k_base", 8191),
}


class OpenAIEmbedder(TextEmbedder):
    """Embedder via OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int | None = None) -> None:
        """Setup.
        model and dim: see https://platform.openai.com/docs/models/embeddings
        `text-embedding-3-large` model: max 3072 dim
        `text-embedding-3-small` model: max 1536 dim
        `text-embedding-ada-002` model: max 1536 dim

        OpenAI embeddings have been normalized to length 1. See
            https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use

        As OpenAI embedding APIs don't allow an empty string as input, we replace an empty string with a
            space " " to avoid error.

        Args:
            model (str, optional): Name of an OpenAI embedding model. Defaults to "text-embedding-3-small".
            dim (int, optional): Output dimension. Defaults to 1536.
        """
        self.client = OpenAI()  # API key written in .env
        assert model in OPENAI_EMBEDDERS.keys(), f"`model` must be one of {list(OPENAI_EMBEDDERS.keys())}!"
        self.model = model
        model_config = OPENAI_EMBEDDERS[model]
        self.encoding = tiktoken.get_encoding(model_config.encoder_name)
        self.max_token_length = model_config.max_token_length
        if not dim or model == "text-embedding-ada-002":
            self.dim = model_config.max_output_dim
        else:
            if dim > model_config.max_output_dim:
                self.dim = model_config.max_output_dim
                logger.warning(f"The maximum dimension of model {self.model} is {self.dim}, use dim={self.dim}.")
            else:
                self.dim = dim

    def encode(self, text: str | list[str], prefix: str | None = None) -> np.ndarray:
        kwargs = {"dimensions": self.dim} if self.model != "text-embedding-ada-002" else {}
        # specifying `dimensions` is not allowed for "text-embedding-ada-002"
        if isinstance(text, str):
            token_ids: list[int] = self.encode_and_truncate_text(text)
        else:
            token_ids: list[list[int]] = [self.encode_and_truncate_text(t) for t in text]
        result = np.asarray(
            [
                data.embedding
                for data in self.client.embeddings.create(
                    input=token_ids,
                    model=self.model,
                    **kwargs,
                ).data
            ]
        )
        if result.shape[0] == 1:
            return result.reshape(-1)
        return result

    def get_output_dim(self) -> int:
        return self.dim

    def encode_and_truncate_text(self, text: str, prefix: str | None = None) -> list[int]:
        # Refer to https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
        # return a list of token IDs
        if not text:
            text = " "
            logger.warning("Found empty string!")
        return self.encoding.encode(text, prefix)[: self.max_token_length]
