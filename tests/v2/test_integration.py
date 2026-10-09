"""
Smoke tests running real MTEB evaluations (no mocks) with a tiny model and small tasks.

They download the model and datasets from the Hugging Face Hub, and are meant to catch
regressions in the JMTEBModel -> mteb.evaluate integration (e.g. after updating mteb).
"""

import json
import math

import pytest
from sentence_transformers import SentenceTransformer

from jmteb.embedders import SentenceBertEmbedder
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks

MODEL_NAME_OR_PATH = "prajjwal1/bert-tiny"
# Small tasks covering both MTEB's encode-only path (STS) and its search path (Retrieval)
TASK_NAMES = ["JSTS", "NLPJournalTitleAbsRetrieval.V2"]


class CustomModel:
    """Custom model as in the README, only encoding a plain list of texts."""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME_OR_PATH)

    def encode(self, sentences, batch_size=32, **kwargs):
        return self.model.encode(sentences, batch_size=batch_size)


def evaluate(model, task_names, tmp_path):
    save_path = tmp_path / "results"
    evaluator = JMTEBV2Evaluator(
        model=model,
        tasks=get_jmteb_tasks(task_names=task_names),
        save_path=save_path,
        cache_path=tmp_path / "cache",
    )
    results = evaluator.run()
    summary = json.loads((save_path / "summary.json").read_text())
    return results, summary


def test_evaluate_from_sentence_transformer(tmp_path):
    model = JMTEBModel.from_sentence_transformer(
        MODEL_NAME_OR_PATH,
        prompts={"Retrieval-query": "query: ", "Retrieval-document": "passage: "},
    )
    results, summary = evaluate(model, TASK_NAMES, tmp_path)

    assert [result.task_results[0].task_name for result in results] == TASK_NAMES
    for category, task_key, main_metric in [
        ("STS", "jsts", "cosine_spearman"),
        ("Retrieval", "nlp_journal_title_abs", "ndcg_at_10"),
    ]:
        entry = summary[category][task_key]
        assert entry["main_metric"] == main_metric
        assert math.isfinite(entry["main_score"])
    # Per-task results are cached by MTEB
    assert len(list((tmp_path / "cache").glob("results/*/*/JSTS.json"))) == 1


@pytest.mark.parametrize(
    "build_model",
    [
        pytest.param(lambda: JMTEBModel(sentence_transformer=CustomModel(), model_name="test/custom"), id="custom"),
        pytest.param(
            lambda: JMTEBModel.from_jmteb_embedder(SentenceBertEmbedder(MODEL_NAME_OR_PATH), model_name="test/v1"),
            id="v1_embedder",
        ),
    ],
)
def test_text_encoder_models_match_sentence_transformer(build_model, tmp_path):
    """Models wrapped by TextEncoderWrapper give the same scores as the SentenceTransformer path."""
    expected, _ = evaluate(JMTEBModel.from_sentence_transformer(MODEL_NAME_OR_PATH), ["JSTS"], tmp_path / "st")
    results, summary = evaluate(build_model(), ["JSTS"], tmp_path / "text_encoder")

    expected_score = expected[0].task_results[0].scores["validation"][0]["main_score"]
    score = results[0].task_results[0].scores["validation"][0]["main_score"]
    assert score == pytest.approx(expected_score, abs=1e-6)
    assert summary["STS"]["jsts"]["main_score"] == pytest.approx(expected_score * 100, abs=1e-4)
