from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from loguru import logger

from .helper import get_similarities


class ThresholdF1Metric:
    def evaluate(
        self,
        embeddings1: Union[np.ndarray, torch.Tensor],
        embeddings2: Union[np.ndarray, torch.Tensor],
        golden: List[int],
    ) -> Dict:
        n_class = len(set(golden))

        if n_class != 2:
            raise ValueError("Support only binary classification.")

        similarities = get_similarities(embeddings1, embeddings2)

        scores: dict[str, float] = {}
        for dist_metric, dist in similarities.items():
            high_score_more_similar = True if dist_metric in ["cosine_distance", "dot_similarities"] else False
            f1s: Dict[str, Tuple] = self._find_best_f1_threshold_binary(dist, golden, high_score_more_similar)[0]
            logger.info(f1s)
            for average_method, f1_p_r in f1s.items():
                _scores = {}
                metric_name = f"{average_method}_f1"
                _scores[metric_name] = f1_p_r[0]
            scores[dist_metric] = _scores
        return scores

    @staticmethod
    def _find_best_f1_threshold_binary(
        scores: np.ndarray,
        labels: List,
        high_score_more_similar: bool,
        reverse_labels: bool = False,
    ) -> Tuple[Dict[str, float], float]:
        """Find the threshold that induces the best F1.
        Assume a label is either 0 or 1.

        Args:
            scores (np.ndarray): similarity/distance scores
            labels (List): labels
            high_score_more_similar (bool): set True if higher score means higher similarity.
            reverse_labels (bool): reverse 0 and 1 to correctly calculate precision, recall and F1.
                Defaults to False.

        Returns:
            Tuple[Dict[str, float], float]:
                (best f1, precision, recall) and the best threshold.
        """
        assert len(scores) == len(labels)
        rows = list(zip(scores, [1 - label for label in labels] if reverse_labels else labels))
        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_precision = best_recall = best_f1 = 0

        best_threshold = -1
        positive_so_far = 0
        remaining_negatives = sum(np.array(labels) == 0)
        total_num_positive = sum(labels)

        for i in range(len(rows) - 1):
            label = rows[i][1]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            precision = positive_so_far / (i + 1)
            recall = positive_so_far / total_num_positive
            f1 = 2 / (1 / precision + 1 / recall) if positive_so_far > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return {"binary": (best_f1, best_precision, best_recall)}, best_threshold
