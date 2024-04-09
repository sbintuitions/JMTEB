from __future__ import annotations

import numpy as np
from loguru import logger
from openai import OpenAI

from jmteb.embedders.base import TextEmbedder

MODEL_DIM = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
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

        Args:
            model (str, optional): Name of an OpenAI embedding model. Defaults to "text-embedding-3-small".
            dim (int, optional): Output dimension. Defaults to 1536.
        """
        self.client = OpenAI()  # API key written in .env
        assert model in MODEL_DIM.keys(), f"`model` must be one of {list(MODEL_DIM.keys())}!"
        self.model = model
        if not dim:
            self.dim = MODEL_DIM[self.model]
        else:
            if dim > MODEL_DIM[self.model]:
                self.dim = MODEL_DIM[self.model]
                logger.warning(f"The maximum dimension of model {self.model} is {self.dim}, use dim={self.dim}.")
            else:
                self.dim = dim

    def encode(self, text: str | list[str]) -> np.ndarray:
        result = np.asarray(
            [
                data.embedding
                for data in self.client.embeddings.create(
                    input=text,
                    model=self.model,
                    dimensions=self.dim,
                ).data
            ]
        )
        if result.shape[0] == 1:
            return result.reshape(-1)
        return result

    def get_output_dim(self) -> int:
        return self.dim
