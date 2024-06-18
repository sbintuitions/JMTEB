from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import PairClassificationDataset
from .threshold_accuracy import ThresholdAccuracyMetric
from .threshold_f1 import ThresholdF1Metric


class PairClassificationEvaluator(EmbeddingEvaluator):
    """
    Evaluator for pair classification task.

    Args:
        val_dataset (PairClassificationDataset): validation dataset
        test_dataset (PairClassificationDataset): test dataset
        sentence1_prefix (str | None): prefix for sentence1. Defaults to None.
        sentence2_prefix (str | None): prefix for sentence2. Defaults to None.
    """

    def __init__(
        self,
        val_dataset: PairClassificationDataset,
        test_dataset: PairClassificationDataset,
        sentence1_prefix: str | None = None,
        sentence2_prefix: str | None = None,
    ) -> None:
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.sentence1_prefix = sentence1_prefix
        self.sentence2_prefix = sentence2_prefix
        self.metrics = [ThresholdAccuracyMetric(), ThresholdF1Metric()]
        self.main_metric = "binary_f1"

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        val_embeddings1, val_embeddings2, val_golden_labels = self._convert_to_embeddings(
            model, self.val_dataset, "dev", overwrite_cache, cache_dir
        )
        if self.val_dataset == self.test_dataset:
            test_embeddings1, test_embeddings2, test_golden_labels = (
                val_embeddings1,
                val_embeddings2,
                val_golden_labels,
            )
        else:
            test_embeddings1, test_embeddings2, test_golden_labels = self._convert_to_embeddings(
                model, self.test_dataset, "test", overwrite_cache, cache_dir
            )

        val_results = {}
        test_results = {}

        for metric in self.metrics:
            _val_results = metric.evaluate(
                embeddings1=val_embeddings1,
                embeddings2=val_embeddings2,
                golden=val_golden_labels,
            )  # {dist_metric: {metric_name: score, metric_name_threshold: threshold}}
            for k, v in _val_results.items():
                if k not in val_results:
                    val_results[k] = v
                else:
                    val_results[k].update(v)

        sorted_val_results = sorted(
            val_results.items(),
            key=lambda res: res[1][self.main_metric],
            reverse=True,
        )

        optimal_dist_metric = sorted_val_results[0][0]
        # keys = ["binary_f1", "binary_f1_threshold", "accuracy", "accuracy_threshold"]
        optimal_thresholds = {k: v for k, v in sorted_val_results[0][1].items() if k.endswith("_threshold")}

        for metric in self.metrics:
            _test_results = metric.evaluate(
                embeddings1=test_embeddings1,
                embeddings2=test_embeddings2,
                golden=test_golden_labels,
                dist_metric=optimal_dist_metric,
                thresholds=optimal_thresholds,
            )
            for k, v in _test_results.items():
                if k not in test_results:
                    test_results[k] = v
                else:
                    test_results[k].update(v)

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=max([v[self.main_metric] for v in test_results.values()]),
            details={
                "optimal_distance_metric": optimal_dist_metric,
                "val_scores": val_results,
                "test_scores": test_results,
            },
        )

    def _convert_to_embeddings(
        self,
        model: TextEmbedder,
        dataset: PairClassificationDataset,
        split: str = "test",
        overwrite_cache: bool = False,
        cache_dir: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        embeddings1 = model.batch_encode_with_cache(
            [item.sentence1 for item in dataset],
            prefix=self.sentence1_prefix,
            cache_path=Path(cache_dir) / f"{split}_embeddings1.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        embeddings2 = model.batch_encode_with_cache(
            [item.sentence2 for item in dataset],
            prefix=self.sentence2_prefix,
            cache_path=Path(cache_dir) / f"{split}_embeddings2.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        golden_labels = [item.label for item in dataset]
        return embeddings1, embeddings2, golden_labels
