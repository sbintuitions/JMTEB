from __future__ import annotations

from copy import deepcopy
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
        if self.val_dataset == self.test_dataset:
            test_embeddings = val_embeddings
            test_labels = val_labels
        else:
            test_embeddings = model.batch_encode_with_cache(
                [item.text for item in self.test_dataset],
                cache_path=Path(cache_dir) / "test_embeddings.bin" if cache_dir is not None else None,
                overwrite_cache=overwrite_cache,
            )
            test_labels = [item.label for item in self.test_dataset]

        n_clusters = len(set(test_labels))
        clustering_models = {
            type(model).__name__: model
            for model_constructor in (
                lambda: MiniBatchKMeans(n_clusters=n_clusters, n_init="auto"),
                lambda: AgglomerativeClustering(n_clusters=n_clusters),
                lambda: BisectingKMeans(n_clusters=n_clusters),
                lambda: Birch(n_clusters=n_clusters),
            )
        }

        logger.info("Fitting clustering model...")
        val_results = {}
        for clustering_model_name, clustering_model in clustering_models.items():
            val_results[clustering_model_name] = self._evaluate_clustering_model(
                val_embeddings, val_labels, clustering_model
            )
        optimal_clustering_model_name = sorted(
            val_results.items(),
            key=lambda res: res[1][self.main_metric],
            reverse=True,
        )[0][0]

        test_results = {
            optimal_clustering_model_name: self._evaluate_clustering_model(
                test_embeddings,
                test_labels,
                clustering_models[optimal_clustering_model_name],
            )
        }

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=test_results[optimal_clustering_model_name][self.main_metric],
            details={
                "optimal_clustering_model_name": optimal_clustering_model_name,
                "val_scores": val_results,
                "test_scores": test_results,
            },
        )

    @staticmethod
    def _evaluate_clustering_model(embeddings: np.ndarray, y_true: list, model_constructor: Callable[[], ClusteringModel]) -> dict[str, float]:
        clustering_model_ = model_constructor()
        clustering_model_.fit(embeddings)
        y_pred = clustering_model_.labels_
        h_score, c_score, v_score = homogeneity_completeness_v_measure(
            labels_pred=y_pred, labels_true=np.array(y_true)
        )
        del clustering_model_
        return {
            "v_measure_score": v_score,
            "homogeneity_score": h_score,
            "completeness_score": c_score,
        }
