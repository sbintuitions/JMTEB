from __future__ import annotations

from os import PathLike
from pathlib import Path

from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .classifiers import Classifier, KnnClassifier, LogRegClassifier
from .data import ClassificationDataset


class ClassificationEvaluator(EmbeddingEvaluator):
    """
    Evaluator for classification task.

    Args:
        train_dataset (ClassificationDataset): training dataset
        dev_dataset (ClassificationDataset): validation dataset
        test_dataset (ClassificationDataset): evaluation dataset
        average (str): average method used in multiclass classification in F1 score and average precision score,
            One of `micro`, `macro`, `samples`, `weighted`, `binary`. Multiple average methods are allowed,
            and delimited by comma, e.g., `macro, micro`.
            The first one is specified as the main index.
        classifiers (dict[str, Classifier]): classifiers to be evaluated.
    """

    def __init__(
        self,
        train_dataset: ClassificationDataset,
        dev_dataset: ClassificationDataset,
        test_dataset: ClassificationDataset,
        average: str = "macro",
        classifiers: dict[str, Classifier] | None = None,
    ) -> None:
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.classifiers = classifiers or {
            "knn_cosine_k_2": KnnClassifier(k=2, distance_metric="cosine"),
            "logreg": LogRegClassifier(),
        }
        self.average = [
            average_name.strip().lower()
            for average_name in average
            if average_name.strip().lower() in ("micro", "macro", "samples", "weighted", "binary")
        ] or ["macro"]
        self.main_metric = f"{self.average[0]}_f1"

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Encoding training and validation sentences...")
        X_train = model.batch_encode_with_cache(
            [item.text for item in self.train_dataset],
            cache_path=Path(cache_dir) / "train_embeddings.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        y_train = [item.label for item in self.train_dataset]

        X_dev = model.batch_encode_with_cache(
            [item.text for item in self.dev_dataset],
            cache_path=Path(cache_dir) / "dev_embeddings.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        y_dev = [item.label for item in self.dev_dataset]

        logger.info("Encoding test sentences...")
        X_test = model.batch_encode_with_cache(
            [item.text for item in self.test_dataset],
            cache_path=Path(cache_dir) / "test_embeddings.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        y_test = [item.label for item in self.test_dataset]

        test_results: dict[str, float] = {}
        dev_results: dict[str, float] = {}
        for classifier_name, classifier in self.classifiers.items():
            logger.info(f"Fitting classifier {classifier_name}...")
            classifier.fit(X_train, y_train)
            logger.info("Evaluating...")

            if self.dev_dataset:
                y_dev_pred = classifier.predict(X_dev)
                dev_results[classifier_name] = self._compute_metrics(y_dev_pred, y_dev, self.average)

        sorted_dev_results = sorted(
            dev_results.items(),
            key=lambda res: res[1][self.main_metric],
            reverse=True,
        )
        optimal_classifier_name = sorted_dev_results[0][0]

        optimal_classifier = self.classifiers[optimal_classifier_name]
        y_pred = optimal_classifier.predict(X_test)
        test_results[optimal_classifier_name] = self._compute_metrics(y_pred, y_test, self.average)

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=test_results[optimal_classifier_name][self.main_metric],
            details={
                "optimal_classifier_name": optimal_classifier_name,
                "dev_scores": dev_results,
                "test_scores": test_results,
            },
        )

    @staticmethod
    def _compute_metrics(y_pred: list, y_true: list, average: list) -> dict[str, float]:
        classifier_results = {}
        classifier_results["accuracy"] = accuracy_score(y_true, y_pred)
        for average_method in average:
            classifier_results[f"{average_method}_f1"] = f1_score(y_true, y_pred, average=average_method)
        return classifier_results
