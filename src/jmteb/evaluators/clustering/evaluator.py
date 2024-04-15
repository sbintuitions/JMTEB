from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    BisectingKMeans,
    MiniBatchKMeans,
)
from sklearn.metrics import homogeneity_completeness_v_measure

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import ClusteringDataset


class ClusteringEvaluator(EmbeddingEvaluator):
    """
    ClusteringEvaluator is a class for evaluating clustering models.
    """

    def __init__(
        self,
        val_dataset: ClusteringDataset,
        test_dataset: ClusteringDataset,
    ) -> None:
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.main_metric = "v_measure_score"

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Converting validation data to embeddings...")
        val_embeddings = model.batch_encode_with_cache(
            [item.text for item in self.val_dataset],
            cache_path=Path(cache_dir) / "val_embeddings.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        val_labels = [item.label for item in self.val_dataset]

        logger.info("Converting test data to embeddings...")
        test_embeddings = model.batch_encode_with_cache(
            [item.text for item in self.test_dataset],
            cache_path=Path(cache_dir) / "test_embeddings.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        test_labels = [item.label for item in self.test_dataset]

        n_clusters = len(set(test_labels))
        clustering_models = {
            type(model).__name__: model
            for model in (
                MiniBatchKMeans(n_clusters=n_clusters, n_init="auto"),
                AgglomerativeClustering(n_clusters=n_clusters),
                BisectingKMeans(n_clusters=n_clusters),
                Birch(n_clusters=n_clusters),
            )
        }

        logger.info("Fitting clustering model...")
        val_results = self._evaluate_clustering_models(val_embeddings, val_labels, clustering_models)
        optimal_clustering_model_name = sorted(
            val_results.items(),
            key=lambda res: res[1][self.main_metric],
            reverse=True,
        )[0][0]

        test_results = self._evaluate_clustering_models(
            test_embeddings,
            test_labels,
            {optimal_clustering_model_name: clustering_models[optimal_clustering_model_name]},
        )

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=test_results[optimal_clustering_model_name][self.main_metric],
            details={
                "optimal_clustering_model_name": optimal_clustering_model_name,
                "val_scores": val_results,
                "test_scores": test_results,
            },
        )

    def _evaluate_clustering_models(
        self, embeddings: np.ndarray, y_true: list, clustering_models: dict[str, Any]
    ) -> dict[str, float]:
        results = {}
        for clustering_model_name, clustering_model in clustering_models.items():
            clustering_model.fit(embeddings)
            y_pred = clustering_model.labels_
            h_score, c_score, v_score = homogeneity_completeness_v_measure(
                labels_pred=y_pred, labels_true=np.array(y_true)
            )
            results[clustering_model_name] = {
                "v_measure_score": v_score,
                "homogeneity_score": h_score,
                "completeness_score": c_score,
            }
        return results
