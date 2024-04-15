from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import Tensor

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import STSDataset


class STSEvaluator(EmbeddingEvaluator):
    """
    Evaluator for STS task.

    Args:
        val_dataset (STSDataset): dev dataset for hyperparameter tuning
        test_dataset (STSDataset): test dataset
    """

    def __init__(self, val_dataset: STSDataset, test_dataset: STSDataset) -> None:
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.main_metric = "spearman"

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        val_embeddings1, val_embeddings2, val_golden_scores = self._convert_to_embeddings(
            model, self.val_dataset, "dev", overwrite_cache, cache_dir
        )
        test_embeddings1, test_embeddings2, test_golden_scores = self._convert_to_embeddings(
            model, self.test_dataset, "test", overwrite_cache, cache_dir
        )

        val_results = {}
        test_results = {}

        similarity_metrics = {
            "cosine_similarity": PairwiseSimilarities.cosine_similarity,
            "manhatten_distance": PairwiseSimilarities.negative_manhatten_distance,
            "euclidean_distance": PairwiseSimilarities.negative_euclidean_distance,
            "dot_score": PairwiseSimilarities.dot_score,
        }

        val_results = self._compute_similarity_scores(
            val_embeddings1, val_embeddings2, val_golden_scores, similarity_metrics
        )
        optimal_similarity_metric = sorted(
            val_results.items(),
            key=lambda res: res[1][self.main_metric],
            reverse=True,
        )[0][0]
        test_results = self._compute_similarity_scores(
            test_embeddings1,
            test_embeddings2,
            test_golden_scores,
            {optimal_similarity_metric: similarity_metrics[optimal_similarity_metric]},
        )

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=test_results[optimal_similarity_metric][self.main_metric],
            details={
                "optimal_similarity_metric": optimal_similarity_metric,
                "val_scores": val_results,
                "test_scores": test_results,
            },
        )

    @staticmethod
    def _compute_similarity_scores(
        embeddings1: Tensor,
        embeddings2: Tensor,
        golden_scores: list,
        similarity_metrics: dict[str, callable],
    ) -> dict[str, dict[str, float]]:
        results = {}
        for dist_name, dist_func in similarity_metrics.items():
            test_sim_score = dist_func(embeddings1, embeddings2).cpu()
            results[dist_name] = {
                "pearson": pearsonr(golden_scores, test_sim_score)[0],
                "spearman": spearmanr(golden_scores, test_sim_score)[0],
            }
        return results

    @staticmethod
    def _convert_to_embeddings(
        model: TextEmbedder,
        dataset: STSDataset,
        split: str = "test",
        overwrite_cache: bool = False,
        cache_dir: str | None = None,
    ) -> tuple[Tensor, Tensor, list[float]]:
        embeddings1 = model.batch_encode_with_cache(
            [item.sentence1 for item in dataset],
            cache_path=Path(cache_dir) / f"{split}_embeddings1.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        embeddings2 = model.batch_encode_with_cache(
            [item.sentence2 for item in dataset],
            cache_path=Path(cache_dir) / f"{split}_embeddings2.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings1 = convert_to_tensor(embeddings1, device)
        embeddings2 = convert_to_tensor(embeddings2, device)

        golden_scores = [item.score for item in dataset]
        return embeddings1, embeddings2, golden_scores


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
