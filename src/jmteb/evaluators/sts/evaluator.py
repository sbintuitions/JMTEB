from __future__ import annotations

import math
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import Tensor

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import STSDataset, STSInstance, STSPrediction


class STSEvaluator(EmbeddingEvaluator):
    """
    Evaluator for STS task.

    Args:
        val_dataset (STSDataset): dev dataset for hyperparameter tuning
        test_dataset (STSDataset): test dataset
        sentence1_prefix (str | None): prefix for sentence1. Defaults to None.
        sentence2_prefix (str | None): prefix for sentence2. Defaults to None.
        encode_kwargs (dict): kwargs passed to embedder's encode function. Defaults to {}.
    """

    def __init__(
        self,
        val_dataset: STSDataset,
        test_dataset: STSDataset,
        sentence1_prefix: str | None = None,
        sentence2_prefix: str | None = None,
        log_predictions: bool = False,
        encode_kwargs: dict = {},
    ) -> None:
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.sentence1_prefix = sentence1_prefix
        self.sentence2_prefix = sentence2_prefix
        self.main_metric = "spearman"
        self.log_predictions = log_predictions
        self.encode_kwargs = encode_kwargs

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        val_embeddings1, val_embeddings2, val_golden_scores = self._convert_to_embeddings(
            model, self.val_dataset, "dev", overwrite_cache, cache_dir
        )
        if self.val_dataset == self.test_dataset:
            test_embeddings1, test_embeddings2, test_golden_scores = (
                val_embeddings1,
                val_embeddings2,
                val_golden_scores,
            )
        test_embeddings1, test_embeddings2, test_golden_scores = self._convert_to_embeddings(
            model, self.test_dataset, "test", overwrite_cache, cache_dir
        )

        similarity_functions = {
            "cosine_similarity": PairwiseSimilarities.cosine_similarity,
            "manhatten_distance": PairwiseSimilarities.negative_manhatten_distance,
            "euclidean_distance": PairwiseSimilarities.negative_euclidean_distance,
            "dot_score": PairwiseSimilarities.dot_score,
        }

        val_results = {}
        for sim_name, sim_func in similarity_functions.items():
            val_results[sim_name], _ = self._compute_similarity(
                val_embeddings1, val_embeddings2, val_golden_scores, sim_func
            )

        optimal_similarity_name = sorted(
            val_results.items(),
            key=lambda res: res[1][self.main_metric],
            reverse=True,
        )[
            0
        ][0]
        test_eval_scores, test_sim_scores = self._compute_similarity(
            test_embeddings1,
            test_embeddings2,
            test_golden_scores,
            similarity_functions[optimal_similarity_name],
        )

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=test_eval_scores[self.main_metric],
            details={
                "optimal_similarity_metric": optimal_similarity_name,
                "val_scores": val_results,
                "test_scores": {optimal_similarity_name: test_eval_scores},
            },
            predictions=(
                self._format_predictions(self.test_dataset, test_sim_scores, optimal_similarity_name)
                if self.log_predictions
                else None
            ),
        )

    @staticmethod
    def _compute_similarity(
        embeddings1: Tensor, embeddings2: Tensor, golden_scores: list, similarity_func: Callable
    ) -> tuple[dict[str, float], list[float]]:
        sim_scores = similarity_func(embeddings1, embeddings2).cpu()
        if isinstance(sim_scores, Tensor) and sim_scores.dtype is torch.bfloat16:
            sim_scores = sim_scores.float()
        pearson = pearsonr(golden_scores, sim_scores)[0]
        spearman = spearmanr(golden_scores, sim_scores)[0]
        return {
            "pearson": pearson if not math.isnan(pearson) else 0.0,
            "spearman": spearman if not math.isnan(spearman) else 0.0,
        }, sim_scores.tolist()

    @staticmethod
    def _format_predictions(
        dataset: STSDataset, sim_scores: list[float], similarity_function_name: str
    ) -> list[STSPrediction]:
        predictions = []
        for item, sim_score in zip(dataset, sim_scores):
            item: STSInstance
            predictions.append(
                STSPrediction(
                    sentence1=item.sentence1,
                    sentence2=item.sentence2,
                    true_score=item.score,
                    predicted_score=sim_score,
                    similarity_function_name=similarity_function_name,
                )
            )
        return predictions

    def _convert_to_embeddings(
        self,
        model: TextEmbedder,
        dataset: STSDataset,
        split: str = "test",
        overwrite_cache: bool = False,
        cache_dir: str | None = None,
    ) -> tuple[Tensor, Tensor, list[float]]:
        embeddings1 = model.batch_encode_with_cache(
            [item.sentence1 for item in dataset],
            prefix=self.sentence1_prefix,
            cache_path=Path(cache_dir) / f"{split}_embeddings1.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
            **self.encode_kwargs,
        )
        embeddings2 = model.batch_encode_with_cache(
            [item.sentence2 for item in dataset],
            prefix=self.sentence2_prefix,
            cache_path=Path(cache_dir) / f"{split}_embeddings2.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
            **self.encode_kwargs,
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
