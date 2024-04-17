from jmteb.evaluators.pair_classification import (
    PairClassificationDataset,
    PairClassificationEvaluator,
    PairClassificationInstance,
)
from jmteb.evaluators.pair_classification.data import JsonlPairClassificationDataset

EXPECTED_OUTPUT_DICT_KEYS = {"val_scores", "test_scores", "optimal_distance_metric"}
EXPECTED_METRIC_NAMES = {"accuracy", "binary_f1", "accuracy_threshold", "binary_f1_threshold"}
EXPECTED_DIST_FUNC_NAMES = {"cosine_distances", "dot_similarities", "manhatten_distances", "euclidean_distances"}


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
    evaluator = PairClassificationEvaluator(val_dataset=DummyBinaryDataset(), test_dataset=DummyBinaryDataset())
    results = evaluator(model=embedder)

    assert results.metric_name in EXPECTED_METRIC_NAMES
    assert set(results.details.keys()) == EXPECTED_OUTPUT_DICT_KEYS
    assert results.details["optimal_distance_metric"] in EXPECTED_DIST_FUNC_NAMES
    assert set(results.details["val_scores"].keys()) == EXPECTED_DIST_FUNC_NAMES
    for value in results.details["val_scores"].values():
        assert set(value.keys()) == EXPECTED_METRIC_NAMES
    assert len(results.details["test_scores"].keys()) == 1
    assert list(results.details["test_scores"].keys())[0] in EXPECTED_DIST_FUNC_NAMES
    for value in results.details["test_scores"].values():
        assert set(value.keys()) == EXPECTED_METRIC_NAMES


def test_pair_classification_jsonl_dataset():
    dataset = JsonlPairClassificationDataset(
        filename="tests/test_data/dummy_pair_classification/binary.jsonl",
        sentence1_key="sentence1",
        sentence2_key="sentence2",
        label_key="label",
    )
    assert dataset.dataset.features["label"].dtype.startswith("int")
    assert len(dataset) == 50


def test_pair_classification_jsonl_dataset_equal():
    dataset_1 = JsonlPairClassificationDataset(
        filename="tests/test_data/dummy_pair_classification/binary.jsonl",
        sentence1_key="sentence1",
        sentence2_key="sentence2",
        label_key="label",
    )
    dataset_2 = JsonlPairClassificationDataset(
        filename="tests/test_data/dummy_pair_classification/binary.jsonl",
        sentence1_key="sentence1",
        sentence2_key="sentence2",
        label_key="label",
    )
    assert dataset_1 == dataset_2
    dataset_2.label_key = "LABEL"
    assert dataset_1 != dataset_2
