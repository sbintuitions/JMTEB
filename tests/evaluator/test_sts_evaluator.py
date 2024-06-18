from jmteb.evaluators.sts import STSDataset, STSEvaluator, STSInstance
from jmteb.evaluators.sts.data import JsonlSTSDataset

EXPECTED_OUTPUT_DICT_KEYS = {"val_scores", "test_scores", "optimal_similarity_metric"}
EXPECTED_SIM_FUNC_NAMES = {"cosine_similarity", "manhatten_distance", "euclidean_distance", "dot_score"}
EXPECTED_METRIC_NAMES = {"pearson", "spearman"}
SENT1_PREFIX = "文1: "
SENT2_PREFIX = "文2: "


class DummySTSDataset(STSDataset):

    def __init__(self, sent1_prefix: str = "", sent2_prefix: str = ""):
        self._items = [
            STSInstance(f"{sent1_prefix}dummy sentence 1", f"{sent2_prefix}dummy sentence 2", i * 0.3)
            for i in range(10)
        ]

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


def test_sts_with_prefix(embedder):
    evaluator_with_prefix = STSEvaluator(
        val_dataset=DummySTSDataset(),
        test_dataset=DummySTSDataset(),
        sentence1_prefix=SENT1_PREFIX,
        sentence2_prefix=SENT2_PREFIX,
    )
    evaluator_with_manual_prefix = STSEvaluator(
        val_dataset=DummySTSDataset(sent1_prefix=SENT1_PREFIX, sent2_prefix=SENT2_PREFIX),
        test_dataset=DummySTSDataset(sent1_prefix=SENT1_PREFIX, sent2_prefix=SENT2_PREFIX),
    )
    assert evaluator_with_prefix(embedder) == evaluator_with_manual_prefix(embedder)


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
