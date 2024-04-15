from jmteb.evaluators.reranking import (
    RerankingDoc,
    RerankingDocDataset,
    RerankingEvaluator,
    RerankingQuery,
    RerankingQueryDataset,
)
from jmteb.evaluators.reranking.data import (
    JsonlRerankingDocDataset,
    JsonlRerankingQueryDataset,
)


class DummyDocDataset(RerankingDocDataset):
    def __init__(self):
        self._items = [RerankingDoc(id=str(i), text=f"dummy document {i}") for i in range(30)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class DummyQueryDataset(RerankingQueryDataset):
    def __init__(self):
        self._items = [
            RerankingQuery(query=f"dummy query {i}", retrieved_docs=[str(i)], relevance_scores=[1]) for i in range(10)
        ]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_reranking_evaluator(embedder):
    evaluator = RerankingEvaluator(
        val_query_dataset=DummyQueryDataset(),
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
    )
    results = evaluator(model=embedder)
    expected_distance_metrics = {"cosine_similarity", "euclidean_distance", "dot_score"}

    assert results.metric_name == "ndcg@10"
    assert set(results.details.keys()) == {"val_scores", "test_scores", "optimal_distance_metric"}
    assert results.details["optimal_distance_metric"] in expected_distance_metrics
    assert set(results.details["val_scores"].keys()) == expected_distance_metrics
    assert list(results.details["test_scores"].keys()) in [[sim] for sim in expected_distance_metrics]
    for score_splitname in ("val_scores", "test_scores"):
        for scores in results.details[score_splitname].values():
            for score in scores.keys():
                assert any(score.startswith(metric) for metric in ["ndcg"])


def test_jsonl_reranking_datasets():
    query = JsonlRerankingQueryDataset(
        filename="tests/test_data/dummy_reranking/val.jsonl",
    )
    assert len(query) == 10

    corpus = JsonlRerankingDocDataset(
        filename="tests/test_data/dummy_reranking/corpus.jsonl",
    )
    assert len(corpus) == 10


def test_jsonl_reranking_datasets_equal():
    query_1 = JsonlRerankingQueryDataset(filename="tests/test_data/dummy_reranking/val.jsonl")
    query_2 = JsonlRerankingQueryDataset(filename="tests/test_data/dummy_reranking/val.jsonl")
    assert query_1 == query_2

    corpus_1 = JsonlRerankingDocDataset(filename="tests/test_data/dummy_reranking/corpus.jsonl")
    corpus_2 = JsonlRerankingDocDataset(filename="tests/test_data/dummy_reranking/corpus.jsonl")
    assert corpus_1 == corpus_2
