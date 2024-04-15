from jmteb.evaluators.sts import STSDataset, STSEvaluator, STSInstance
from jmteb.evaluators.sts.data import JsonlSTSDataset


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

    expected_metrics = {"pearson", "spearman"}
    expected_sims = {"cosine_similarity", "manhatten_distance", "euclidean_distance", "dot_score"}

    assert results.metric_name in expected_metrics
    assert set(results.details.keys()) == {"dev_scores", "test_scores", "optimal_similarity_metric"}
    assert results.details["optimal_similarity_metric"] in expected_sims
    assert set(results.details["dev_scores"].keys()) == expected_sims
    assert list(results.details["test_scores"].keys()) in [[dist] for dist in expected_sims]
    for score_splitname in ("dev_scores", "test_scores"):
        for dist in expected_sims:
            if dist in results.details[score_splitname]:
                assert set(results.details[score_splitname][dist].keys()) == expected_metrics


def test_sts_jsonl_dataset():
    dataset = JsonlSTSDataset(
        filename="tests/test_data/dummy_sts/dev.jsonl",
        sentence1_key="sentence1",
        sentence2_key="sentence2",
        label_key="label",
    )
    assert len(dataset) == 50
