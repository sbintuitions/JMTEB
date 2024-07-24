from jmteb.evaluators.classification import (
    ClassificationDataset,
    ClassificationEvaluator,
    ClassificationInstance,
    KnnClassifier,
    LogRegClassifier,
)
from jmteb.evaluators.classification.data import JsonlClassificationDataset

EXPECTED_OUTPUT_DICT_KEYS = {"val_scores", "test_scores", "optimal_classifier_name"}
PREFIX = "以下の文を分類する: "


class DummyClassificationDataset(ClassificationDataset):
    def __init__(self, prefix: str = ""):
        self._items = [ClassificationInstance(text=f"{prefix}dummy text {i}", label=i // 2) for i in range(10)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_classification_evaluator(embedder):
    evaluator = ClassificationEvaluator(
        train_dataset=DummyClassificationDataset(),
        val_dataset=DummyClassificationDataset(),
        test_dataset=DummyClassificationDataset(),
        classifiers={
            "logreg": LogRegClassifier(),
            "knn": KnnClassifier(k=2, distance_metric="cosine"),
        },
    )
    results = evaluator(model=embedder)
    expected_metrics = {"accuracy", "macro_f1"}
    assert results.metric_name in expected_metrics
    assert set(results.details.keys()) == EXPECTED_OUTPUT_DICT_KEYS
    assert results.details["optimal_classifier_name"] in {"logreg", "knn"}
    assert set(results.details["val_scores"].keys()) == {"logreg", "knn"}
    assert list(results.details["test_scores"].keys()) in (["logreg"], ["knn"])
    for score_splitname in ("val_scores", "test_scores"):
        for value in results.details[score_splitname].values():
            assert set(value.keys()) == expected_metrics


def test_classification_evaluator_with_prefix(embedder):
    evaluator_with_prefix = ClassificationEvaluator(
        train_dataset=DummyClassificationDataset(),
        val_dataset=DummyClassificationDataset(),
        test_dataset=DummyClassificationDataset(),
        prefix=PREFIX,
        classifiers={
            "logreg": LogRegClassifier(),
            "knn": KnnClassifier(k=2, distance_metric="cosine"),
        },
    )
    evaluator_with_manual_prefix = ClassificationEvaluator(
        train_dataset=DummyClassificationDataset(prefix=PREFIX),
        val_dataset=DummyClassificationDataset(prefix=PREFIX),
        test_dataset=DummyClassificationDataset(prefix=PREFIX),
        classifiers={
            "logreg": LogRegClassifier(),
            "knn": KnnClassifier(k=2, distance_metric="cosine"),
        },
    )
    assert evaluator_with_prefix(embedder) == evaluator_with_manual_prefix(embedder)


def test_classification_jsonl_dataset():
    dummy_jsonl_dataset = JsonlClassificationDataset(
        filename="tests/test_data/dummy_classification/val.jsonl",
        text_key="sentence",
        label_key="label",
    )
    assert len(dummy_jsonl_dataset) == 10


def test_classification_jsonl_dataset_equal():
    dummy_jsonl_dataset_1 = JsonlClassificationDataset(
        filename="tests/test_data/dummy_classification/val.jsonl",
        text_key="sentence",
        label_key="label",
    )
    dummy_jsonl_dataset_2 = JsonlClassificationDataset(
        filename="tests/test_data/dummy_classification/val.jsonl",
        text_key="sentence",
        label_key="label",
    )
    assert dummy_jsonl_dataset_1 == dummy_jsonl_dataset_2
    dummy_jsonl_dataset_2.label_key = "LABEL"
    assert dummy_jsonl_dataset_1 != dummy_jsonl_dataset_2


def test_classification_prediction_logging(embedder):
    dataset = DummyClassificationDataset()
    evaluator = ClassificationEvaluator(
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        classifiers={
            "logreg": LogRegClassifier(),
            "knn": KnnClassifier(k=2, distance_metric="cosine"),
        },
        log_predictions=True,
    )
    results = evaluator(model=embedder)
    assert isinstance(results.predictions, list)
    assert [p.text for p in results.predictions] == [d.text for d in dataset]
    assert [p.label for p in results.predictions] == [d.label for d in dataset]
    assert all([isinstance(p.prediction, int) for p in results.predictions])
