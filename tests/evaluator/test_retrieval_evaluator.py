from jmteb.evaluators.retrieval import (
    RetrievalDoc,
    RetrievalDocDataset,
    RetrievalEvaluator,
    RetrievalQuery,
    RetrievalQueryDataset,
)
from jmteb.evaluators.retrieval.data import (
    JsonlRetrievalDocDataset,
    JsonlRetrievalQueryDataset,
)


class DummyDocDataset(RetrievalDocDataset):
    def __init__(self):
        self._items = [RetrievalDoc(id=str(i), text=f"dummy document {i}") for i in range(30)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class DummyQueryDataset(RetrievalQueryDataset):
    def __init__(self):
        self._items = [RetrievalQuery(f"dummy query {i}", relevant_docs=[str(i)]) for i in range(10)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_retrieval_evaluator(embedder):
    evaluator = RetrievalEvaluator(
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        accuracy_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5],
        doc_chunk_size=3,
    )
    results = evaluator(model=embedder)
    expected_distance_metrics = {"cosine_similarity", "euclidean_distance", "dot_score"}

    assert results.metric_name == "ndcg@1"
    assert set(results.details.keys()) == {"dev_scores", "test_scores", "optimal_distance_metric"}
    assert results.details["optimal_distance_metric"] in expected_distance_metrics
    assert results.details["dev_scores"] == {}
    assert set(results.details["test_scores"].keys()) == expected_distance_metrics
    for scores in results.details["test_scores"].values():
        for score in scores.keys():
            assert any(score.startswith(metric) for metric in ["accuracy", "mrr", "ndcg"])


def test_retrieval_evaluator_with_hyperparameter_tuning(embedder):
    evaluator = RetrievalEvaluator(
        test_query_dataset=DummyQueryDataset(),
        dev_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        accuracy_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5],
        doc_chunk_size=3,
    )
    results = evaluator(model=embedder)
    expected_distance_metrics = {"cosine_similarity", "euclidean_distance", "dot_score"}

    assert results.metric_name == "ndcg@1"
    assert set(results.details.keys()) == {"dev_scores", "test_scores", "optimal_distance_metric"}
    assert results.details["optimal_distance_metric"] in expected_distance_metrics
    for score_splitname in ("dev_scores", "test_scores"):
        assert set(results.details[score_splitname].keys()) == expected_distance_metrics
        for scores in results.details[score_splitname].values():
            for score in scores.keys():
                assert any(score.startswith(metric) for metric in ["accuracy", "mrr", "ndcg"])


def test_if_chunking_does_not_change_result(embedder):
    evaluator1 = RetrievalEvaluator(
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        doc_chunk_size=3,
    )

    evaluator2 = RetrievalEvaluator(
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        doc_chunk_size=30,
    )

    assert evaluator1(embedder) == evaluator2(embedder)


def test_jsonl_retrieval_datasets():
    query = JsonlRetrievalQueryDataset(
        filename="tests/test_data/dummy_retrieval/dev.jsonl",
        query_key="question",
        relevant_docs_key="answer",
    )
    assert len(query) == 10

    corpus = JsonlRetrievalDocDataset(
        filename="tests/test_data/dummy_retrieval/corpus.jsonl",
        id_key="docid",
        text_key="text",
    )
    assert len(corpus) == 10
