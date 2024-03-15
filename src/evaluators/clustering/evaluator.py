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

from src.embedders.base import TextEmbedder
from src.evaluators.base import EmbeddingEvaluator, EvaluationResults

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
        dataset: ClusteringDataset,
        k: int = 2,
        k_means_clustering: bool = True,
        hierarchical_clustering: bool = False,
    ) -> None:
        self.dataset = dataset
        self.k = k
        self.k_means_clustering = k_means_clustering
        self.hierarchical_clustering = hierarchical_clustering
        self.main_metric = "v_measure_score"

    def __call__(
        self, model: TextEmbedder, cache_dir: str | PathLike[str] | None = None, overwrite_cache: bool = False
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        embeddings = model.batch_encode_with_cache(
            [item.text for item in self.dataset],
            cache_path=Path(cache_dir) / "embeddings.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        labels = [item.label for item in self.dataset]
        n_clusters = len(set(labels))

        results: dict[str, dict[str, float]] = {}

        logger.info("Fitting clustering model...")
        clustering_models = []
        if self.k_means_clustering:
            clustering_models.append(
                MiniBatchKMeans(
                    n_clusters=n_clusters,
                    n_init="auto",
                )
            )
        if self.hierarchical_clustering:
            clustering_models.extend(
                [
                    AgglomerativeClustering(n_clusters=n_clusters),
                    BisectingKMeans(n_clusters=n_clusters),
                    Birch(n_clusters=n_clusters),
                ]
            )
        for clustering_model in clustering_models:
            _results: dict[str, float] = {}
            clustering_model.fit(embeddings)
            y_pred = clustering_model.labels_

            # compute metric
            h_score, c_score, v_score = homogeneity_completeness_v_measure(
                labels_pred=y_pred, labels_true=np.array(labels)
            )
            clustering_model_name = type(clustering_model).__name__
            _results.update(
                {
                    "v_measure_score": v_score,
                    "homogeneity_score": h_score,
                    "completeness_score": c_score,
                }
            )
            results[clustering_model_name] = _results

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=max([v[self.main_metric] for v in results.values()]),
            details=results,
        )
