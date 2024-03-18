from src.evaluators.pair_classification import (
    PairClassificationDataset,
    PairClassificationEvaluator,
    PairClassificationInstance,
)
from src.evaluators.pair_classification.data import JsonlPairClassificationDataset


class DummyBinaryDataset(PairClassificationDataset):
    def __init__(self):
        self._items = [
            PairClassificationInstance(f"dummy sentence 1 {i}", f"dummy sentence 2 {i}", i % 2) for i in range(10)
        ]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_pair_classification_binary(embedder):
    evaluator = PairClassificationEvaluator(
        dataset=DummyBinaryDataset(),
    )
    results = evaluator(model=embedder)

    expected_metrics = {"accuracy", "binary_f1"}
    expected_distances = {"cosine_distances", "dot_similarities", "manhatten_distances", "euclidean_distances"}

    assert results.metric_name in expected_metrics
    assert set(results.details.keys()) == expected_distances
    for value in results.details.values():
        assert set(value.keys()) == expected_metrics


def test_pair_classification_jsonl_dataset():
    dataset = JsonlPairClassificationDataset(
        filename="tests/test_data/dummy_pair_classification/binary.jsonl",
        sentence1_key="sentence1",
        sentence2_key="sentence2",
        label_key="label",
    )
    assert dataset.dataset.features["label"].dtype.startswith("int")
    assert len(dataset) == 50
