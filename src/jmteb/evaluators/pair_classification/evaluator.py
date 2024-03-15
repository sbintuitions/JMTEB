from __future__ import annotations

from os import PathLike
from pathlib import Path

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import PairClassificationDataset
from .threshold_accuracy import ThresholdAccuracyMetric
from .threshold_f1 import ThresholdF1Metric


class PairClassificationEvaluator(EmbeddingEvaluator):
    """
    Evaluator for pair classification task.

    Args:
        dataset (PairClassificationDataset): dataset
    """

    def __init__(self, dataset: PairClassificationDataset) -> None:
        self.dataset = dataset
        self.metrics = [ThresholdAccuracyMetric(), ThresholdF1Metric()]
        self.main_metric = "binary_f1"

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

        results: dict[str, float] = {}
        for metric in self.metrics:
            _results = metric.evaluate(
                embeddings1=embeddings1,
                embeddings2=embeddings2,
                golden=[item.label for item in self.dataset],
            )
            for k, v in _results.items():
                if k not in results:
                    results[k] = v
                else:
                    results[k].update(v)

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=max([v[self.main_metric] for v in results.values()]),
            details=results,
        )
