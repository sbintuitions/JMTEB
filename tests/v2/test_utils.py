"""
Tests for JMTEB v2.0 utility functions.
"""

import json

from jmteb.v2 import utils


class TestUtils:
    """Tests for utility functions."""

    def test_load_prompts(self, tmp_path):
        """Test loading prompts from YAML file."""
        prompt_file = tmp_path / "prompts.yaml"
        prompt_file.write_text("query: 'query: '\npassage: 'passage: '\n")

        prompts = utils.load_prompts(prompt_file)

        assert prompts["query"] == "query: "
        assert prompts["passage"] == "passage: "

    def test_load_prompts_empty(self, tmp_path):
        """Test loading empty prompts file."""
        prompt_file = tmp_path / "empty.yaml"
        prompt_file.write_text("")

        prompts = utils.load_prompts(prompt_file)

        assert prompts == {}

    def test_load_batch_sizes(self, tmp_path):
        """Test loading batch sizes from YAML file."""
        batch_file = tmp_path / "batch_sizes.yaml"
        batch_file.write_text("JSTS: 128\nJSICK: 64\nJaqketRetrieval: 32\n")

        batch_sizes = utils.load_batch_sizes(batch_file)

        assert batch_sizes["JSTS"] == 128
        assert batch_sizes["JSICK"] == 64
        assert batch_sizes["JaqketRetrieval"] == 32

    def test_load_summary_exists(self, tmp_path):
        """Test loading existing summary.json."""
        summary_file = tmp_path / "summary.json"
        summary_data = {"STS": {"jsts": {"main_score": 82.14}}}
        summary_file.write_text(json.dumps(summary_data))

        summary = utils.load_summary(tmp_path)

        assert summary == summary_data

    def test_load_summary_not_exists(self, tmp_path):
        """Test loading summary when file doesn't exist."""
        summary = utils.load_summary(tmp_path)

        assert summary == {}

    def test_save_summary(self, tmp_path):
        """Test saving summary to file."""
        summary_data = {"STS": {"jsts": {"main_score": 82.14}}}

        utils.save_summary(summary_data, tmp_path)

        summary_file = tmp_path / "summary.json"
        assert summary_file.exists()

        loaded = json.loads(summary_file.read_text())
        assert loaded == summary_data

    def test_extract_and_update_summary_jsts(self, tmp_path):
        """Test extracting and updating summary for JSTS task."""
        # Create result file
        result_file = tmp_path / "JSTS.json"
        result_data = {"validation": [{"main_score": 0.8234, "cosine_spearman": 0.8234}]}
        result_file.write_text(json.dumps(result_data))

        summary = {}
        utils.extract_and_update_summary(
            task_name="JSTS",
            main_metric="cosine_spearman",
            save_path=tmp_path,
            summary=summary,
            eval_time=5.67,
        )

        assert "STS" in summary
        assert "jsts" in summary["STS"]
        assert summary["STS"]["jsts"]["main_score"] == 82.34  # 0.8234 * 100
        assert summary["STS"]["jsts"]["eval_time (s)"] == "5.67"

    def test_extract_and_update_summary_mldr(self, tmp_path):
        """Test extracting and updating summary for MLDR task (uses dev split)."""
        # Create result file
        result_file = tmp_path / "MultiLongDocRetrieval.json"
        result_data = {"dev": [{"main_score": 0.7512, "ndcg@10": 0.7512}]}
        result_file.write_text(json.dumps(result_data))

        summary = {}
        utils.extract_and_update_summary(
            task_name="MultiLongDocRetrieval",
            main_metric="ndcg@10",
            save_path=tmp_path,
            summary=summary,
            eval_time=120.5,
        )

        assert "Retrieval" in summary
        assert "mldr_retrieval" in summary["Retrieval"]
        assert summary["Retrieval"]["mldr_retrieval"]["main_score"] == 75.12

    def test_extract_and_update_summary_cached(self, tmp_path):
        """Test updating summary for cached result."""
        result_file = tmp_path / "JSICK.json"
        result_data = {"test": [{"main_score": 0.7689}]}
        result_file.write_text(json.dumps(result_data))

        summary = {}
        utils.extract_and_update_summary(
            task_name="JSICK",
            main_metric="cosine_spearman",
            save_path=tmp_path,
            summary=summary,
            eval_time=-1,  # Cached
        )

        assert summary["STS"]["jsick"]["eval_time (s)"] == "cached"

    def test_extract_and_update_summary_unknown_task(self, tmp_path):
        """Test that unknown task is skipped."""
        summary = {}
        utils.extract_and_update_summary(
            task_name="UnknownTask",
            main_metric="some_metric",
            save_path=tmp_path,
            summary=summary,
            eval_time=10.0,
        )

        # Summary should remain empty
        assert summary == {}

    def test_save_results(self, tmp_path):
        """Test saving generic results."""
        results = {"task1": {"score": 0.85}, "task2": {"score": 0.92}}

        utils.save_results(results, tmp_path, filename="custom_results.json")

        result_file = tmp_path / "custom_results.json"
        assert result_file.exists()

        loaded = json.loads(result_file.read_text())
        assert loaded == results

    def test_get_task_key_mapping(self):
        """Test internal _get_task_key function."""
        # Test some common mappings
        assert utils._get_task_key("JSTS") == "jsts"
        assert utils._get_task_key("JaqketRetrieval") == "jaqket"
        assert utils._get_task_key("LivedoorNewsClustering.v2") == "livedoor_news"
        assert utils._get_task_key("AmazonReviewsClassification") == "amazon_review_classification"
        assert utils._get_task_key("MultiLongDocRetrieval") == "mldr_retrieval"

        # Unknown task should be lowercased
        assert utils._get_task_key("UnknownTask") == "unknowntask"
