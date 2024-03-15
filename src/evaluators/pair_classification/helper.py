from typing import Dict, Union

import numpy as np
import torch
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from torch import Tensor


def get_similarities(
    embeddings1: Union[np.ndarray, Tensor],
    embeddings2: Union[np.ndarray, Tensor],
) -> Dict[str, np.ndarray]:
    cosine_distances = 1 - paired_cosine_distances(embeddings1, embeddings2)
    manhatten_distances = paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

    batch_size, dim = embeddings1.shape
    dot_similarities = (
        torch.bmm(
            torch.tensor(embeddings1).view(batch_size, 1, dim),
            torch.tensor(embeddings2).view(batch_size, dim, 1),
        )
        .view(-1)
        .numpy()
    )
    return {
        "cosine_distances": cosine_distances,
        "manhatten_distances": manhatten_distances,
        "euclidean_distances": euclidean_distances,
        "dot_similarities": dot_similarities,
    }
