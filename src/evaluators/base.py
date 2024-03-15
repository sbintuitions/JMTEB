from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from typing import Any

from pydantic.dataclasses import dataclass

from src.embedders import TextEmbedder


@dataclass
class EvaluationResults:
    """
    Evaluation results.

    Args:
        metric_name (str): Name of the metric. This is the primary metric to compare models.
        metric_value (float): Value of the main metric.
        details (dict[str, Any]): Details of the evaluation.
            This included some additional metrics or values that are used to derive the main metric.
    """

    metric_name: str
    metric_value: float
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "details": self.details,
        }


class EmbeddingEvaluator(ABC):
    """Abstract evaluator class."""

    @abstractmethod
    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        raise NotImplementedError
