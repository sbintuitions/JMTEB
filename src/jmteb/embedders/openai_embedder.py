from __future__ import annotations

import numpy as np
from openai import OpenAI

from jmteb.embedders.base import TextEmbedder


class OpenAIEmbedder(TextEmbedder):
    """Embedder via OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int | None = None) -> None:
        """Setup.
        model and dim: see https://platform.openai.com/docs/models/embeddings
        `text-embedding-3-large` model: max 3072 dim
        `text-embedding-3-small` model: max 1536 dim
        `text-embedding-ada-002` model: max 1536 dim

        Args:
            model (str, optional): Name of an OpenAI embedding model. Defaults to "text-embedding-3-small".
            dim (int, optional): Output dimension. Defaults to 1536.
        """
        self.client = OpenAI()  # API key written in .env
        self.model = model
        if not dim:
            if model == "text-embedding-3-large":
                self.dim = 3072
            else:
                self.dim = 1536
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
