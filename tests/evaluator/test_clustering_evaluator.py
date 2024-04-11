from jmteb.evaluators.clustering import (
    ClusteringDataset,
    ClusteringEvaluator,
    ClusteringInstance,
)
from jmteb.evaluators.clustering.data import JsonlClusteringDataset


class DummyClusteringDataset(ClusteringDataset):
    def __init__(self):
        self._items = [ClusteringInstance(text=f"dummy text {i}", label=i // 2) for i in range(10)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_kmeans_clustering(embedder):
    evaluator = ClusteringEvaluator(test_dataset=DummyClusteringDataset())
    results = evaluator(model=embedder)
    expected_metrics = {"v_measure_score", "completeness_score", "homogeneity_score"}
    assert results.metric_name in expected_metrics
    assert set(results.details.keys()) == {"dev_scores", "test_scores", "optimal_clustering_model_name"}
    expected_clustering_models = {"MiniBatchKMeans", "AgglomerativeClustering", "BisectingKMeans", "Birch"}
    assert set(results.details["test_scores"].keys()) == expected_clustering_models
    for clustering_model in expected_clustering_models:
        assert set(results.details["test_scores"][clustering_model].keys()) == expected_metrics


def test_clustering_jsonl_dataset():
    dataset = JsonlClusteringDataset(
        filename="tests/test_data/dummy_clustering/dev.jsonl",
        text_key="text",
        label_key="label",
    )
    assert len(dataset) == 50
