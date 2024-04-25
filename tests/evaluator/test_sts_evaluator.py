from jmteb.evaluators.sts import STSDataset, STSEvaluator, STSInstance
from jmteb.evaluators.sts.data import JsonlSTSDataset

EXPECTED_OUTPUT_DICT_KEYS = {"val_scores", "test_scores", "optimal_similarity_metric"}
EXPECTED_SIM_FUNC_NAMES = {"cosine_similarity", "manhatten_distance", "euclidean_distance", "dot_score"}
EXPECTED_METRIC_NAMES = {"pearson", "spearman"}


class DummySTSDataset(STSDataset):
    def __init__(self):
        self._items = [STSInstance("dummy sentence 1", "dummy sentence 2", i * 0.3) for i in range(10)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_sts(embedder):
    evaluator = STSEvaluator(val_dataset=DummySTSDataset(), test_dataset=DummySTSDataset())
    results = evaluator(model=embedder)

    assert results.metric_name in EXPECTED_METRIC_NAMES
    assert set(results.details.keys()) == EXPECTED_OUTPUT_DICT_KEYS
    assert results.details["optimal_similarity_metric"] in EXPECTED_SIM_FUNC_NAMES
    assert set(results.details["val_scores"].keys()) == EXPECTED_SIM_FUNC_NAMES
    assert list(results.details["test_scores"].keys()) in [[dist] for dist in EXPECTED_SIM_FUNC_NAMES]
    for score_splitname in ("val_scores", "test_scores"):
        for dist in EXPECTED_SIM_FUNC_NAMES:
            if dist in results.details[score_splitname]:
                assert set(results.details[score_splitname][dist].keys()) == EXPECTED_METRIC_NAMES


def test_sts_jsonl_dataset():
    dataset = JsonlSTSDataset(
        filename="tests/test_data/dummy_sts/val.jsonl",
        sentence1_key="sentence1",
        sentence2_key="sentence2",
        label_key="label",
    )
    assert len(dataset) == 50


def test_sts_jsonl_dataset_equal():
    dataset_1 = JsonlSTSDataset(
        filename="tests/test_data/dummy_sts/val.jsonl",
        sentence1_key="sentence1",
        sentence2_key="sentence2",
        label_key="label",
    )
    dataset_2 = JsonlSTSDataset(
        filename="tests/test_data/dummy_sts/val.jsonl",
        sentence1_key="sentence1",
        sentence2_key="sentence2",
        label_key="label",
    )
    assert dataset_1 == dataset_2
    dataset_2.label_key = "LABEL"
    assert dataset_1 != dataset_2
