from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger
from sklearn.base import ClusterMixin
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    BisectingKMeans,
    MiniBatchKMeans,
)
from sklearn.metrics import homogeneity_completeness_v_measure

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import ClusteringDataset, ClusteringPrediction


class ClusteringEvaluator(EmbeddingEvaluator):
    """
    ClusteringEvaluator is a class for evaluating clustering models.
    """

    def __init__(
        self,
        val_dataset: ClusteringDataset,
        test_dataset: ClusteringDataset,
        prefix: str | None = None,
        random_seed: int | None = None,
        log_predictions: bool = False,
        encode_kwargs: dict = {},
    ) -> None:
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.prefix = prefix
        self.random_seed = random_seed
        self.log_predictions = log_predictions
        self.encode_kwargs = encode_kwargs
        self.main_metric = "v_measure_score"

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Converting validation data to embeddings...")
        val_embeddings = model.batch_encode_with_cache(
            [item.text for item in self.val_dataset],
            prefix=self.prefix,
            cache_path=Path(cache_dir) / "val_embeddings.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
            **self.encode_kwargs,
        )
        val_labels = [item.label for item in self.val_dataset]

        logger.info("Converting test data to embeddings...")
        if self.val_dataset == self.test_dataset:
            test_embeddings = val_embeddings
            test_labels = val_labels
        else:
            test_embeddings = model.batch_encode_with_cache(
                [item.text for item in self.test_dataset],
                prefix=self.prefix,
                cache_path=Path(cache_dir) / "test_embeddings.bin" if cache_dir is not None else None,
                overwrite_cache=overwrite_cache,
                **self.encode_kwargs,
            )
            test_labels = [item.label for item in self.test_dataset]

        n_clusters = len(set(test_labels))
        model_constructors: dict[str, Callable[[], ClusterMixin]] = {
            "MiniBatchKMeans": lambda: MiniBatchKMeans(
                n_clusters=n_clusters, n_init="auto", random_state=self.random_seed
            ),
            "AgglomerativeClustering": lambda: AgglomerativeClustering(n_clusters=n_clusters),
            "BisectingKMeans": lambda: BisectingKMeans(n_clusters=n_clusters, random_state=self.random_seed),
            "Birch": lambda: Birch(n_clusters=n_clusters),
        }

        logger.info("Fitting clustering model...")
        val_results = {}
        for model_name, model_constructor in model_constructors.items():
            val_results[model_name], _ = self._evaluate_clustering_model(
                val_embeddings, val_labels, model_constructor()
            )
        optimal_clustering_model_name = sorted(
            val_results.items(),
            key=lambda res: res[1][self.main_metric],
            reverse=True,
        )[0][0]

        test_scores, test_predictions = self._evaluate_clustering_model(
            test_embeddings,
            test_labels,
            model_constructors[optimal_clustering_model_name](),
        )
        test_results = {optimal_clustering_model_name: test_scores}

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=test_results[optimal_clustering_model_name][self.main_metric],
            details={
                "optimal_clustering_model_name": optimal_clustering_model_name,
                "val_scores": val_results,
                "test_scores": test_results,
            },
            predictions=(
                self._format_predictions(self.test_dataset, test_predictions) if self.log_predictions else None
            ),
        )

    @staticmethod
    def _evaluate_clustering_model(
        embeddings: np.ndarray, y_true: list[int], clustering_model: ClusterMixin
    ) -> tuple[dict[str, float], list[int]]:
        y_pred = clustering_model.fit_predict(embeddings)
        h_score, c_score, v_score = homogeneity_completeness_v_measure(
            labels_pred=y_pred, labels_true=np.array(y_true)
        )
        del clustering_model
        return {
            "v_measure_score": v_score,
            "homogeneity_score": h_score,
            "completeness_score": c_score,
        }, y_pred.tolist()

    @staticmethod
    def _format_predictions(dataset: ClusteringDataset, predictions: list[int]) -> list[ClusteringPrediction]:
        return [
            ClusteringPrediction(item.text, item.label, prediction) for item, prediction in zip(dataset, predictions)
        ]
