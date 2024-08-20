from jmteb.evaluators.retrieval import (
    RetrievalDoc,
    RetrievalDocDataset,
    RetrievalEvaluator,
    RetrievalPrediction,
    RetrievalQuery,
    RetrievalQueryDataset,
)
from jmteb.evaluators.retrieval.data import (
    JsonlRetrievalDocDataset,
    JsonlRetrievalQueryDataset,
)

EXPECTED_OUTPUT_DICT_KEYS = {"val_scores", "test_scores", "optimal_distance_metric"}
EXPECTED_DIST_FUNC_NAMES = {"cosine_similarity", "euclidean_distance", "dot_score"}
QUERY_PREFIX = "クエリ: "
DOC_PREFIX = "ドキュメント: "
TOP_N_DOCS_TO_LOG = 4


class DummyDocDataset(RetrievalDocDataset):
    def __init__(self, prefix: str = ""):
        self._items = [RetrievalDoc(id=str(i), text=f"{prefix}dummy document {i}") for i in range(30)]
        self._build_idx_docid_mapping("_items")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class DummyQueryDataset(RetrievalQueryDataset):
    def __init__(self, prefix: str = ""):
        self._items = [RetrievalQuery(f"{prefix}dummy query {i}", relevant_docs=[str(i)]) for i in range(10)]

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


def test_retrieval_evaluator_with_predictions(embedder):
    dummy_query_dataset = DummyQueryDataset()
    dummy_doc_dataset = DummyDocDataset()
    evaluator = RetrievalEvaluator(
        val_query_dataset=dummy_query_dataset,
        test_query_dataset=dummy_query_dataset,
        doc_dataset=dummy_doc_dataset,
        accuracy_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5],
        doc_chunk_size=3,
        log_predictions=True,
        top_n_docs_to_log=TOP_N_DOCS_TO_LOG,
    )
    results = evaluator(model=embedder)
    assert [p.query for p in results.predictions] == [q.query for q in dummy_query_dataset]
    assert all([isinstance(p, RetrievalPrediction) for p in results.predictions])
    for p in results.predictions:
        assert isinstance(p, RetrievalPrediction)
        assert len(p.predicted_relevant_docs) == TOP_N_DOCS_TO_LOG
        assert all([isinstance(doc, RetrievalDoc) for doc in p.predicted_relevant_docs])


def test_retrieval_evaluator_with_prefix(embedder):
    evaluator_with_prefix = RetrievalEvaluator(
        val_query_dataset=DummyQueryDataset(),
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        query_prefix=QUERY_PREFIX,
        doc_prefix=DOC_PREFIX,
        accuracy_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5],
        doc_chunk_size=3,
    )
    evaluator_with_manual_prefix = RetrievalEvaluator(
        val_query_dataset=DummyQueryDataset(prefix=QUERY_PREFIX),
        test_query_dataset=DummyQueryDataset(prefix=QUERY_PREFIX),
        doc_dataset=DummyDocDataset(prefix=DOC_PREFIX),
        accuracy_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5],
        doc_chunk_size=3,
    )
    assert evaluator_with_prefix(model=embedder) == evaluator_with_manual_prefix(model=embedder)


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
