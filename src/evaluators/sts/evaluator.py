from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import Tensor

from src.embedders.base import TextEmbedder
from src.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import STSDataset


class STSEvaluator(EmbeddingEvaluator):
    """
    Evaluator for STS task.

    Args:
        dataset (STSDataset): dataset
    """

    def __init__(self, dataset: STSDataset) -> None:
        self.dataset = dataset
        self.main_metric = "spearman"

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        embeddings1 = model.batch_encode_with_cache(
            [item.sentence1 for item in self.dataset],
            cache_path=Path(cache_dir) / "embeddings1.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        embeddings2 = model.batch_encode_with_cache(
            [item.sentence2 for item in self.dataset],
            cache_path=Path(cache_dir) / "embeddings2.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings1 = convert_to_tensor(embeddings1, device)
        embeddings2 = convert_to_tensor(embeddings2, device)

        golden_scores = [item.score for item in self.dataset]

        scores = {}

        for dist_name, dist_func in {
            "cosine_similarity": PairwiseSimilarities.cosine_similarity,
            "manhatten_distance": PairwiseSimilarities.negative_manhatten_distance,
            "euclidean_distance": PairwiseSimilarities.negative_euclidean_distance,
            "dot_score": PairwiseSimilarities.dot_score,
        }.items():
            sim_score = dist_func(embeddings1, embeddings2).cpu()
            scores[dist_name] = {
                "pearson": pearsonr(golden_scores, sim_score)[0],
                "spearman": spearmanr(golden_scores, sim_score)[0],
            }

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=max([scores[self.main_metric] for scores in scores.values()]),
            details=scores,
        )


@dataclass
class PairwiseSimilarities:
    @staticmethod
    def cosine_similarity(e1: Tensor, e2: Tensor) -> Tensor:
        e1_norm = torch.nn.functional.normalize(e1, p=2, dim=1)
        e2_norm = torch.nn.functional.normalize(e2, p=2, dim=1)
        return (e1_norm * e2_norm).sum(dim=1)

    @staticmethod
    def negative_manhatten_distance(e1: Tensor, e2: Tensor) -> Tensor:
        # the more distant, the less similar, so we use its opposite number
        return -abs(e2 - e1).sum(dim=1)

    @staticmethod
    def negative_euclidean_distance(e1: Tensor, e2: Tensor) -> Tensor:
        # the more distant, the less similar, so we use its opposite number
        return -torch.sqrt(torch.square(e2 - e1)).sum(dim=1)

    @staticmethod
    def dot_score(e1: Tensor, e2: Tensor) -> Tensor:
        return (e1 * e2).sum(dim=1)


def convert_to_tensor(embeddings: np.ndarray | Tensor, device: str) -> Tensor:
    if not isinstance(embeddings, Tensor):
        embeddings = torch.tensor(embeddings)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.unsqueeze(0)
    return embeddings.to(device=device)
