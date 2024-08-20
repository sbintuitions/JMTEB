from loguru import logger

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
    RerankingPrediction,
)

EXPECTED_OUTPUT_DICT_KEYS = {"val_scores", "test_scores", "optimal_distance_metric"}
EXPECTED_DIST_FUNC_NAMES = {"cosine_similarity", "euclidean_distance", "dot_score"}
QUERY_PREFIX = "クエリ: "
DOC_PREFIX = "ドキュメント: "
TOP_N_DOCS_TO_LOG = 4


class DummyDocDataset(RerankingDocDataset):
    def __init__(self, prefix: str = ""):
        self._items = [RerankingDoc(id=str(i), text=f"{prefix}dummy document {i}") for i in range(30)]
        self._build_idx_docid_mapping("_items")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class DummyQueryDataset(RerankingQueryDataset):
    def __init__(self, prefix: str = ""):
        self._items = [
            RerankingQuery(query=f"{prefix}dummy query {i}", retrieved_docs=[str(i)], relevance_scores=[1])
            for i in range(10)
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

    assert results.metric_name == "ndcg@10"
    assert set(results.details.keys()) == EXPECTED_OUTPUT_DICT_KEYS
    assert results.details["optimal_distance_metric"] in EXPECTED_DIST_FUNC_NAMES
    assert set(results.details["val_scores"].keys()) == EXPECTED_DIST_FUNC_NAMES
    assert list(results.details["test_scores"].keys()) in [[sim] for sim in EXPECTED_DIST_FUNC_NAMES]
    for score_splitname in ("val_scores", "test_scores"):
        for scores in results.details[score_splitname].values():
            for score in scores.keys():
                assert any(score.startswith(metric) for metric in ["ndcg"])


def test_reranking_evaluator_with_predictions(embedder):
    evaluator = RerankingEvaluator(
        val_query_dataset=DummyQueryDataset(),
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        log_predictions=True,
        top_n_docs_to_log=TOP_N_DOCS_TO_LOG,
    )
    results = evaluator(model=embedder)
    logger.info(f"{results.predictions=}")
    for p in results.predictions:
        assert isinstance(p, RerankingPrediction)
        assert len(p.reranked_relevant_docs) <= TOP_N_DOCS_TO_LOG
        assert all([isinstance(doc, RerankingDoc) for doc in p.reranked_relevant_docs])


def test_reranking_evaluator_with_prefix(embedder):
    evaluator_with_prefix = RerankingEvaluator(
        val_query_dataset=DummyQueryDataset(),
        test_query_dataset=DummyQueryDataset(),
        doc_dataset=DummyDocDataset(),
        query_prefix=QUERY_PREFIX,
        doc_prefix=DOC_PREFIX,
    )
    evaluator_with_manual_prefix = RerankingEvaluator(
        val_query_dataset=DummyQueryDataset(prefix=QUERY_PREFIX),
        test_query_dataset=DummyQueryDataset(prefix=QUERY_PREFIX),
        doc_dataset=DummyDocDataset(prefix=DOC_PREFIX),
    )
    assert evaluator_with_prefix(model=embedder) == evaluator_with_manual_prefix(model=embedder)


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
    query_2.filename = ""
    assert query_1 != query_2

    corpus_1 = JsonlRerankingDocDataset(filename="tests/test_data/dummy_reranking/corpus.jsonl")
    corpus_2 = JsonlRerankingDocDataset(filename="tests/test_data/dummy_reranking/corpus.jsonl")
    assert corpus_1 == corpus_2
    corpus_2.filename = ""
    assert corpus_1 != corpus_2
