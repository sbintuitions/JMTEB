from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from .helper import get_similarities


class ThresholdAccuracyMetric:
    def evaluate(
        self,
        embeddings1: Union[np.ndarray, torch.Tensor],
        embeddings2: Union[np.ndarray, torch.Tensor],
        golden: List[float],
    ) -> Dict:
        if len(set(golden)) != 2:
            raise ValueError("Support only binary classification.")

        similarities = get_similarities(embeddings1, embeddings2)

        scores: dict[str, float] = {}
        for dist_metric, dist in similarities.items():
            _scores = {}
            high_score_more_similar = True if dist_metric in ["cosine_distance", "dot_similarities"] else False
            accuracy = self._find_best_accuracy_threshold_binary(dist, golden, high_score_more_similar)[0]
            _scores["accuracy"] = accuracy
            scores[dist_metric] = _scores
        return scores

    @staticmethod
    def _find_best_accuracy_threshold_binary(
        scores: np.ndarray,
        labels: List,
        high_score_more_similar: bool,
    ) -> Tuple[float, float]:
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
