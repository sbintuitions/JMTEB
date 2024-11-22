from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import Any

from jmteb.evaluators import EvaluationResults


class AbstractScoreRecorder(ABC):
    @abstractmethod
    def record_task_scores(self, scores: EvaluationResults, dataset_name: str, task_name: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def record_summary(self) -> Any:
        raise NotImplementedError


class JsonScoreRecorder(AbstractScoreRecorder):
    def __init__(self, save_dir: str | None = None) -> None:
        self.save_dir = save_dir
        self.scores: dict[str, dict[str, EvaluationResults]] = defaultdict(dict)

    @staticmethod
    def save_to_json(
        scores: EvaluationResults | dict[Any, Any] | None = None, filename: str | PathLike[str] | None = None
    ) -> None:
        if scores is None or filename is None:
            return
        with open(filename, "w") as fout:
            json.dump(scores, fout, indent=4, ensure_ascii=False)

    @staticmethod
    def save_prediction_to_jsonl(
        predictions: list[Any] | None = None, filename: str | PathLike[str] | None = None
    ) -> None:
        if predictions is None or filename is None:
            return
        with open(filename, "w") as fout:
            for prediction in predictions:
                fout.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")

    def record_task_scores(
        self, scores: EvaluationResults | None = None, dataset_name: str | None = None, task_name: str | None = None
    ) -> None:
        if self.save_dir is None or scores is None or dataset_name is None or task_name is None:
            return
        save_filename = Path(self.save_dir) / task_name / f"scores_{dataset_name}.json"
        save_filename.parent.mkdir(parents=True, exist_ok=True)

        self.scores[task_name][dataset_name] = scores
        self.save_to_json(self.scores[task_name][dataset_name].as_dict(), save_filename)

    def record_predictions(
        self, results: EvaluationResults | None = None, dataset_name: str | None = None, task_name: str | None = None
    ) -> None:
        if not self.save_dir or results is None or dataset_name is None or task_name is None:
            return
        if not hasattr(results, "predictions"):
            return
        save_filename = Path(self.save_dir) / task_name / f"predictions_{dataset_name}.jsonl"
        save_filename.parent.mkdir(parents=True, exist_ok=True)
        self.save_prediction_to_jsonl(results.predictions, save_filename)

    def record_summary(self):
        if not self.save_dir:
            return
        summary: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        for task_name, task_scores in self.scores.items():
            for dataset_name, results in self.scores[task_name].items():
                summary[task_name][dataset_name] = {results.metric_name: results.metric_value}
        self.save_to_json(summary, Path(self.save_dir) / "summary.json")
