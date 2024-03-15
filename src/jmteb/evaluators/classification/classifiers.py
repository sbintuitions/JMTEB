from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class Classifier(ABC):
    """
    Abstract classifier class.
    This is used to perform classification tasks using text embeddings as input features.
    """

    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        pass


class KnnClassifier(Classifier):
    def __init__(self, k: int, distance_metric: str) -> None:
        self._knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric, n_jobs=-1)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._knn.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._knn.predict(features)


class LogRegClassifier(Classifier):
    def __init__(self) -> None:
        self._logreg = LogisticRegression()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._logreg.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._logreg.predict(features)
