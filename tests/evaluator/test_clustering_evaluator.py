from src.evaluators.clustering import (
    ClusteringDataset,
    ClusteringEvaluator,
    ClusteringInstance,
)
from src.evaluators.clustering.data import JsonlClusteringDataset
from tests.evaluator.fixture import embedder  # noqa: F401


class DummyClusteringDataset(ClusteringDataset):
    def __init__(self):
        self._items = [ClusteringInstance(text=f"dummy text {i}", label=i // 2) for i in range(10)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_kmeans_clustering(embedder):
    evaluator = ClusteringEvaluator(dataset=DummyClusteringDataset())
    results = evaluator(model=embedder)
    expected_metrics = {"v_measure_score", "completeness_score", "homogeneity_score"}
    assert results.metric_name in expected_metrics
    assert set(results.details.keys()) == {"MiniBatchKMeans"}
    assert set(results.details["MiniBatchKMeans"].keys()) == expected_metrics


def test_clustering_jsonl_dataset():
    dataset = JsonlClusteringDataset(
        filename="tests/test_data/dummy_clustering/dev.jsonl",
        text_key="text",
        label_key="label",
    )
    assert len(dataset) == 50
