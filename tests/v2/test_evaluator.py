"""
Tests for JMTEB v2.0 evaluator (JMTEBV2Evaluator).
"""

import json
from unittest.mock import Mock, patch

import mteb
import pytest

from jmteb.v2.evaluator import JMTEBV2Evaluator


def make_model_result(scores):
    task_result = Mock()
    task_result.scores = scores
    model_result = Mock()
    model_result.task_results = [task_result]
    return model_result


@pytest.fixture
def tasks():
    # get_tasks returns MTEBTasks, a tuple subclass
    return mteb.get_tasks(tasks=["JSTS", "MIRACLReranking", "AmazonCounterfactualClassification"], languages=["jpn"])


@pytest.fixture
def mock_evaluate():
    scores = {
        "JSTS": {"validation": [{"main_score": 0.8}]},
        "MIRACLReranking": {"dev": [{"main_score": 0.6}]},
        "AmazonCounterfactualClassification": {
            "validation": [{"main_score": 0.1}],
            "test": [{"main_score": 0.7}],
        },
    }
    with patch("jmteb.v2.evaluator.mteb.evaluate") as mock:
        mock.side_effect = lambda model, tasks, **kwargs: make_model_result(scores[tasks.metadata.name])
        yield mock


class TestJMTEBV2Evaluator:
    """Tests for JMTEBV2Evaluator."""

    def test_tasks_tuple_evaluated_one_by_one(self, tasks, mock_evaluate, tmp_path):
        evaluator = JMTEBV2Evaluator(model=Mock(), tasks=tasks, save_path=tmp_path, task_batch_sizes={"JSTS": 128})
        results = evaluator.run()

        assert len(results) == 3
        evaluated = {call.kwargs["tasks"].metadata.name: call.kwargs for call in mock_evaluate.call_args_list}
        assert set(evaluated) == {"JSTS", "MIRACLReranking", "AmazonCounterfactualClassification"}
        assert evaluated["JSTS"]["encode_kwargs"]["batch_size"] == 128
        assert evaluated["MIRACLReranking"]["encode_kwargs"]["batch_size"] == 32
        assert evaluated["JSTS"]["overwrite_strategy"] == "only-missing"

    def test_single_task(self, tasks, mock_evaluate):
        evaluator = JMTEBV2Evaluator(model=Mock(), tasks=tasks[0])
        assert len(evaluator.run()) == 1

    def test_summary(self, tasks, mock_evaluate, tmp_path):
        JMTEBV2Evaluator(model=Mock(), tasks=tasks, save_path=tmp_path).run()

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["STS"]["jsts"]["main_score"] == pytest.approx(80.0)
        assert summary["STS"]["jsts"]["main_metric"] == "cosine_spearman"
        assert summary["Reranking"]["miracl_reranking"]["main_score"] == pytest.approx(60.0)
        assert summary["Classification"]["amazon_counterfactual_classification"]["main_score"] == pytest.approx(70.0)

    def test_overwrite_cache(self, tasks, mock_evaluate):
        JMTEBV2Evaluator(model=Mock(), tasks=tasks, overwrite_cache=True).run()
        assert mock_evaluate.call_args.kwargs["overwrite_strategy"] == "always"

    def test_generate_summary_false(self, tasks, mock_evaluate, tmp_path):
        JMTEBV2Evaluator(model=Mock(), tasks=tasks, save_path=tmp_path, generate_summary=False).run()
        assert not (tmp_path / "summary.json").exists()
