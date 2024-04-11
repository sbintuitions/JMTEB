from __future__ import annotations

from os import PathLike
from pathlib import Path

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

    Args:
        k (int): Number of clusters.
        k_means_clustering (bool): Whether to use K-means clustering. Defaults to True.
        hierarchical_clustering (bool): Whether to use hierarchical clustering methods
            (AgglomerativeClustering, Birch, BisectingKMeans) other than MiniBatchKMeans.
    """

    def __init__(
        self,
        test_dataset: ClusteringDataset,
        dev_dataset: ClusteringDataset | None = None,
    ) -> None:
        self.test_dataset = test_dataset
        self.dev_dataset = dev_dataset
        self.main_metric = "v_measure_score"

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        if self.dev_dataset:
            logger.info("Converting validatin data to embeddings...")
            dev_embeddings = model.batch_encode_with_cache(
                [item.text for item in self.dev_dataset],
                cache_path=Path(cache_dir) / "dev_embeddings.bin" if cache_dir is not None else None,
                overwrite_cache=overwrite_cache,
            )
            dev_labels = [item.label for item in self.dev_dataset]

        logger.info("Converting test data to embeddings...")
        test_embeddings = model.batch_encode_with_cache(
            [item.text for item in self.test_dataset],
            cache_path=Path(cache_dir) / "test_embeddings.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        test_labels = [item.label for item in self.test_dataset]

        logger.info("Fitting clustering model...")
        test_results = self._evaluate_clustering_models(test_embeddings, test_labels)
        dev_results = {}
        if self.dev_dataset:
            dev_results = self._evaluate_clustering_models(dev_embeddings, dev_labels)
        optimal_clustering_model_name = sorted(
            dev_results.items() if dev_results else test_results.items(),
            key=lambda res: res[1][self.main_metric],
            reverse=True,
        )[0][0]

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=test_results[optimal_clustering_model_name][self.main_metric],
            details={
                "optimal_clustering_model_name": optimal_clustering_model_name,
                "dev_scores": dev_results,
                "test_scores": test_results,
            },
        )

    @staticmethod
    def _init_clustering_models(n_clusters):
        return (
            MiniBatchKMeans(n_clusters=n_clusters, n_init="auto"),
            AgglomerativeClustering(n_clusters=n_clusters),
            BisectingKMeans(n_clusters=n_clusters),
            Birch(n_clusters=n_clusters),
        )

    def _evaluate_clustering_models(self, embeddings: np.ndarray, y_true: list) -> dict[str, float]:
        results = {}
        n_clusters = len(set(y_true))
        clustering_models = self._init_clustering_models(n_clusters)
        for clustering_model in clustering_models:
            clustering_model.fit(embeddings)
            y_pred = clustering_model.labels_
            h_score, c_score, v_score = homogeneity_completeness_v_measure(
                labels_pred=y_pred, labels_true=np.array(y_true)
            )
            clustering_model_name = type(clustering_model).__name__
            results[clustering_model_name] = {
                "v_measure_score": v_score,
                "homogeneity_score": h_score,
                "completeness_score": c_score,
            }
        return results
