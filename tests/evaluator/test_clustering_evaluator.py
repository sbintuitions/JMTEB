from jmteb.evaluators.clustering import (
    ClusteringDataset,
    ClusteringEvaluator,
    ClusteringInstance,
)
from jmteb.evaluators.clustering.data import JsonlClusteringDataset

EXPECTED_OUTPUT_DICT_KEYS = {"val_scores", "test_scores", "optimal_clustering_model_name"}
EXPECTED_METRIC_NAMES = {"v_measure_score", "completeness_score", "homogeneity_score"}
EXPECTED_MODEL_NAMES = {"MiniBatchKMeans", "AgglomerativeClustering", "BisectingKMeans", "Birch"}


class DummyClusteringDataset(ClusteringDataset):
    def __init__(self):
        self._items = [ClusteringInstance(text=f"dummy text {i}", label=i // 2) for i in range(10)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_kmeans_clustering(embedder):
    evaluator = ClusteringEvaluator(val_dataset=DummyClusteringDataset(), test_dataset=DummyClusteringDataset())
    results = evaluator(model=embedder)
    expected_metrics = EXPECTED_METRIC_NAMES
    assert results.metric_name in expected_metrics
    assert set(results.details.keys()) == EXPECTED_OUTPUT_DICT_KEYS
    expected_clustering_models = EXPECTED_MODEL_NAMES
    assert results.details["optimal_clustering_model_name"] in expected_clustering_models
    assert set(results.details["val_scores"].keys()) == expected_clustering_models
    assert list(results.details["test_scores"].keys()) in [[model] for model in expected_clustering_models]
    for score_splitname in ("val_scores", "test_scores"):
        for clustering_model in expected_clustering_models:
            if clustering_model in results.details[score_splitname]:
                assert set(results.details[score_splitname][clustering_model].keys()) == expected_metrics


def test_clustering_jsonl_dataset():
    dataset = JsonlClusteringDataset(
        filename="tests/test_data/dummy_clustering/val.jsonl",
        text_key="text",
        label_key="label",
    )
    assert len(dataset) == 50


def test_clustering_jsonl_dataset_equal():
    dataset_1 = JsonlClusteringDataset(
        filename="tests/test_data/dummy_clustering/val.jsonl",
        text_key="text",
        label_key="label",
    )
    dataset_2 = JsonlClusteringDataset(
        filename="tests/test_data/dummy_clustering/val.jsonl",
        text_key="text",
        label_key="label",
    )
    assert dataset_1 == dataset_2
    dataset_2.label_key = "LABEL"
    assert dataset_1 != dataset_2
