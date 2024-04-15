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
        dev_dataset (PairClassificationDataset): validation dataset
        test_dataset (PairClassificationDataset): test dataset
    """

    def __init__(
        self,
        dev_dataset: PairClassificationDataset,
        test_dataset: PairClassificationDataset,
    ) -> None:
        self.test_dataset = test_dataset
        self.dev_dataset = dev_dataset
        self.metrics = [ThresholdAccuracyMetric(), ThresholdF1Metric()]
        self.main_metric = "binary_f1"

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        dev_embeddings1, dev_embeddings2, dev_golden_labels = self._convert_to_embeddings(
            model, self.dev_dataset, "dev", overwrite_cache, cache_dir
        )
        test_embeddings1, test_embeddings2, test_golden_labels = self._convert_to_embeddings(
            model, self.test_dataset, "test", overwrite_cache, cache_dir
        )

        dev_results = {}
        test_results = {}

        for metric in self.metrics:
            _dev_results = metric.evaluate(
                embeddings1=dev_embeddings1,
                embeddings2=dev_embeddings2,
                golden=dev_golden_labels,
            )  # {dist_metric: {metric_name: score, metric_name_threshold: threshold}}
            for k, v in _dev_results.items():
                if k not in dev_results:
                    dev_results[k] = v
                else:
                    dev_results[k].update(v)

        sorted_dev_results = sorted(
            dev_results.items(),
            key=lambda res: res[1][self.main_metric],
            reverse=True,
        )

        optimal_dist_metric = sorted_dev_results[0][0]
        # keys = ["binary_f1", "binary_f1_threshold", "accuracy", "accuracy_threshold"]
        optimal_thresholds = {k: v for k, v in sorted_dev_results[0][1].items() if k.endswith("_threshold")}

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
                "dev_scores": dev_results,
                "test_scores": test_results,
            },
        )

    @staticmethod
    def _convert_to_embeddings(
        model: TextEmbedder,
        dataset: PairClassificationDataset,
        split: str = "test",
        overwrite_cache: bool = False,
        cache_dir: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
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
        golden_labels = [item.label for item in dataset]
        return embeddings1, embeddings2, golden_labels
