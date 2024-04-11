from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from .helper import get_similarities


class ThresholdAccuracyMetric:
    def evaluate(
        self,
        embeddings1: np.ndarray | torch.Tensor,
        embeddings2: np.ndarray | torch.Tensor,
        golden: list[float],
        dist_metric: str | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """evaluate function.
        If `dist_metric` and `threshold` are given (calculated with the dev set), classify with the
        given distance metric and threshold, and compute the accuracy score.
        Otherwise, for all distance metrics, induce the optimal threshold and compute the accuracy.

        Args:
            embeddings1 (np.ndarray | torch.Tensor): embeddings for sentence1
            embeddings2 (np.ndarray | torch.Tensor): embeddings for sentence2
            golden (list[int]): golden labels, 0 or 1
            dist_metric (str | None, optional): distance metric, optimal in dev set. Defaults to None.
            thresholds (dict[str, float] | None, optional): threshold name and value. Defaults to None.

        Raises:
            ValueError: more than binary

        Returns:
            dict[str, dict[str, float]]:
                { dist_metric: {"accuracy": score, "accuracy_threshold": threshold} }
        """

        if len(set(golden)) != 2:
            raise ValueError("Support only binary classification.")

        similarities = get_similarities(embeddings1, embeddings2)

        if dist_metric and thresholds:
            high_score_more_similar = True if dist_metric in ["cosine_distance", "dot_similarities"] else False
            threshold_value = thresholds.get("accuracy_threshold")
            return {
                dist_metric: {
                    "accuracy": self._compute_accuracy_with_given_threshold(
                        similarities[dist_metric], golden, threshold_value, high_score_more_similar
                    ),
                    "accuracy_threshold": threshold_value,
                }
            }

        scores: dict[str, float] = {}
        for dist_metric, dist in similarities.items():
            high_score_more_similar = True if dist_metric in ["cosine_distance", "dot_similarities"] else False
            accuracy, threshold = self._find_best_accuracy_threshold_binary(dist, golden, high_score_more_similar)
            scores[dist_metric] = {"accuracy": accuracy, "accuracy_threshold": threshold}
        return scores

    @staticmethod
    def _find_best_accuracy_threshold_binary(
        scores: np.ndarray,
        labels: list,
        high_score_more_similar: bool,
    ) -> tuple[float, float]:
        """Find the threshold that induces the best accuracy.
        Assume a label is either 0 or 1.

        Args:
            scores (np.ndarray): similarity/distance scores
            labels (List): labels
            high_score_more_similar (bool): set True if higher score means higher similarity.

        Returns:
            Tuple[float, float]:
                max accuracy and the best threshold.
        """
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))
        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1
        positive_so_far = 0
        remaining_negatives = sum(np.array(labels) == 0)

        for i in range(len(rows) - 1):
            label = rows[i][1]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def _compute_accuracy_with_given_threshold(
        similarities: np.ndarray,
        labels: list,
        threshold: float,
        high_score_more_similar: bool,
    ) -> float:
        """Compute accuracy with scores being classified with the given threshold.

        Args:
            similarities (np.ndarray): similarity scores
            labels (list): true labels, 0 or 1
            threshold (float): given threshold
            high_score_more_similar (bool): set True if higher score means higher similarity.

        Returns:
            float: accuarcy
        """
        if high_score_more_similar:
            y_pred = [0 if score < threshold else 1 for score in similarities]
        else:
            y_pred = [1 if score < threshold else 0 for score in similarities]

        return accuracy_score(y_true=labels, y_pred=y_pred)
