from jmteb.evaluators.classification import (
    ClassificationDataset,
    ClassificationEvaluator,
    ClassificationInstance,
    KnnClassifier,
    LogRegClassifier,
)
from jmteb.evaluators.classification.data import JsonlClassificationDataset


class DummyClassificationDataset(ClassificationDataset):
    def __init__(self):
        self._items = [ClassificationInstance(text=f"dummy text {i}", label=i // 2) for i in range(10)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_classification_evaluator(embedder):
    evaluator = ClassificationEvaluator(
        train_dataset=DummyClassificationDataset(),
        test_dataset=DummyClassificationDataset(),
        classifiers={
            "logreg": LogRegClassifier(),
            "knn": KnnClassifier(k=2, distance_metric="cosine"),
        },
    )
    results = evaluator(model=embedder)
    expected_metrics = {"accuracy", "macro_f1"}
    assert results.metric_name in expected_metrics
    assert set(results.details.keys()) == {"dev_scores", "test_scores", "optimal_classifier_name"}
    assert set(results.details["test_scores"].keys()) == {"logreg", "knn"}
    for value in results.details["test_scores"].values():
        assert set(value.keys()) == expected_metrics


def test_classification_jsonl_dataset():
    dummy_jsonl_dataset = JsonlClassificationDataset(
        filename="tests/test_data/dummy_classification/dev.jsonl",
        text_key="sentence",
        label_key="label",
    )
    assert len(dummy_jsonl_dataset) == 10
