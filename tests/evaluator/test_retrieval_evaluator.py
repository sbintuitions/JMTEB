from src.evaluators.retrieval import (
    RetrievalDoc,
    RetrievalDocDataset,
    RetrievalEvaluator,
    RetrievalQuery,
    RetrievalQueryDataset,
)
from src.evaluators.retrieval.data import (
    JsonlRetrievalDocDataset,
    JsonlRetrievalQueryDataset,
)
from tests.evaluator.fixture import embedder  # noqa: F401


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
        query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        accuracy_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5],
        doc_chunk_size=3,
    )
    results = evaluator(model=embedder)

    assert results.metric_name == "ndcg@1"
    assert set(results.details.keys()) == {"cosine_similarity", "euclidean_distance", "dot_score"}
    for scores in results.details.values():
        for score in scores.keys():
            assert any(score.startswith(metric) for metric in ["accuracy", "mrr", "ndcg"])


def test_if_chunking_does_not_change_result(embedder):
    evaluator1 = RetrievalEvaluator(
        query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        doc_chunk_size=3,
    )

    evaluator2 = RetrievalEvaluator(
        query_dataset=DummyQueryDataset(),
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
