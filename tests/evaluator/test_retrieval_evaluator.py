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

EXPECTED_OUTPUT_DICT_KEYS = {"val_scores", "test_scores", "optimal_distance_metric"}
EXPECTED_DIST_FUNC_NAMES = {"cosine_similarity", "euclidean_distance", "dot_score"}


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
        val_query_dataset=DummyQueryDataset(),
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        accuracy_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5],
        doc_chunk_size=3,
    )
    results = evaluator(model=embedder)

    assert results.metric_name == "ndcg@1"
    assert set(results.details.keys()) == EXPECTED_OUTPUT_DICT_KEYS
    assert results.details["optimal_distance_metric"] in EXPECTED_DIST_FUNC_NAMES
    assert set(results.details["val_scores"].keys()) == EXPECTED_DIST_FUNC_NAMES
    assert list(results.details["test_scores"].keys()) in [[sim] for sim in EXPECTED_DIST_FUNC_NAMES]
    for score_splitname in ("val_scores", "test_scores"):
        for scores in results.details[score_splitname].values():
            for score in scores.keys():
                assert any(score.startswith(metric) for metric in ["accuracy", "mrr", "ndcg"])


def test_if_chunking_does_not_change_result(embedder):
    evaluator1 = RetrievalEvaluator(
        val_query_dataset=DummyQueryDataset(),
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        doc_chunk_size=3,
    )

    evaluator2 = RetrievalEvaluator(
        val_query_dataset=DummyQueryDataset(),
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        doc_chunk_size=30,
    )

    assert evaluator1(embedder) == evaluator2(embedder)


def test_jsonl_retrieval_datasets():
    query = JsonlRetrievalQueryDataset(
        filename="tests/test_data/dummy_retrieval/val.jsonl",
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


def test_jsonl_retrieval_datasets_equal():
    query_1 = JsonlRetrievalQueryDataset(
        filename="tests/test_data/dummy_retrieval/val.jsonl",
        query_key="question",
        relevant_docs_key="answer",
    )
    query_2 = JsonlRetrievalQueryDataset(
        filename="tests/test_data/dummy_retrieval/val.jsonl",
        query_key="question",
        relevant_docs_key="answer",
    )
    assert query_1 == query_2
    query_2.relevant_docs_key = "ANSWER"
    assert query_1 != query_2

    corpus_1 = JsonlRetrievalDocDataset(
        filename="tests/test_data/dummy_retrieval/corpus.jsonl",
        id_key="docid",
        text_key="text",
    )
    corpus_2 = JsonlRetrievalDocDataset(
        filename="tests/test_data/dummy_retrieval/corpus.jsonl",
        id_key="docid",
        text_key="text",
    )
    assert corpus_1 == corpus_2
    corpus_2.text_key = "TEXT"
    assert corpus_1 != corpus_2
