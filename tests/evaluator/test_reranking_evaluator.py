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
        self._items = [RerankingQuery(query=f"dummy query {i}", retrieved_docs=[str(i)], relevance_scores=[1]) for i in range(10)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_reranking_evaluator(embedder):
    evaluator = RerankingEvaluator(
        query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
    )
    results = evaluator(model=embedder)

    assert results.metric_name == "ndcg@10"
    assert set(results.details.keys()) == {"cosine_similarity", "euclidean_distance", "dot_score"}
    for scores in results.details.values():
        for score in scores.keys():
            assert any(score.startswith(metric) for metric in ["ndcg"])


def test_jsonl_reranking_datasets():
    query = JsonlRerankingQueryDataset(
        filename="tests/test_data/dummy_reranking/dev.jsonl",
    )
    assert len(query) == 10

    corpus = JsonlRerankingDocDataset(
        filename="tests/test_data/dummy_reranking/corpus.jsonl",
    )
    assert len(corpus) == 10
