"""
Tests for JMTEB v2.0 CLI interface.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from jmteb.v2 import __main__ as cli_module


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_required_model_name(self):
        """Test that model_name is required."""
        with patch.object(sys, "argv", ["prog"]):
            with pytest.raises(SystemExit):
                cli_module.get_args()

    def test_model_name_only(self):
        """Test parsing with only model_name."""
        with patch.object(sys, "argv", ["prog", "--model_name", "cl-nagoya/ruri-v3-30m"]):
            args = cli_module.get_args()
            assert args.model_name == "cl-nagoya/ruri-v3-30m"
            assert args.batch_size == 32  # default
            assert args.fp16 is False  # default
            assert args.bf16 is False  # default

    def test_batch_size_argument(self):
        """Test batch_size argument."""
        with patch.object(
            sys,
            "argv",
            ["prog", "--model_name", "model", "--batch_size", "64"],
        ):
            args = cli_module.get_args()
            assert args.batch_size == 64

    def test_fp16_argument(self):
        """Test fp16 argument."""
        with patch.object(sys, "argv", ["prog", "--model_name", "model", "--fp16", "true"]):
            args = cli_module.get_args()
            assert args.fp16 is True

    def test_bf16_argument(self):
        """Test bf16 argument."""
        with patch.object(sys, "argv", ["prog", "--model_name", "model", "--bf16", "true"]):
            args = cli_module.get_args()
            assert args.bf16 is True

    def test_include_argument(self):
        """Test include argument for task filtering."""
        with patch.object(
            sys,
            "argv",
            ["prog", "--model_name", "model", "--include", '["JSTS", "JSICK"]'],
        ):
            args = cli_module.get_args()
            assert args.include == ["JSTS", "JSICK"]

    def test_exclude_argument(self):
        """Test exclude argument for task filtering."""
        with patch.object(
            sys,
            "argv",
            ["prog", "--model_name", "model", "--exclude", '["JSTS"]'],
        ):
            args = cli_module.get_args()
            assert args.exclude == ["JSTS"]

    def test_task_types_argument(self):
        """Test task_types argument."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--model_name",
                "model",
                "--task_types",
                '["Retrieval", "Classification"]',
            ],
        ):
            args = cli_module.get_args()
            assert args.task_types == ["Retrieval", "Classification"]

    def test_prompt_profile_argument(self):
        """Test prompt_profile argument."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--model_name",
                "model",
                "--prompt_profile",
                "src/jmteb/configs/prompts/e5.yaml",
            ],
        ):
            args = cli_module.get_args()
            assert args.prompt_profile == "src/jmteb/configs/prompts/e5.yaml"

    def test_task_batch_sizes_argument(self):
        """Test task_batch_sizes argument."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--model_name",
                "model",
                "--task_batch_sizes",
                "batch_sizes.yaml",
            ],
        ):
            args = cli_module.get_args()
            assert args.task_batch_sizes == "batch_sizes.yaml"

    def test_save_path_argument(self):
        """Test save_path argument."""
        with patch.object(
            sys,
            "argv",
            ["prog", "--model_name", "model", "--save_path", "my_results"],
        ):
            args = cli_module.get_args()
            assert args.save_path == "my_results"

    def test_overwrite_cache_argument(self):
        """Test overwrite_cache argument."""
        with patch.object(
            sys,
            "argv",
            ["prog", "--model_name", "model", "--overwrite_cache", "true"],
        ):
            args = cli_module.get_args()
            assert args.overwrite_cache is True

    def test_cache_path_argument(self):
        """Test cache_path argument."""
        with patch.object(
            sys,
            "argv",
            ["prog", "--model_name", "model", "--cache_path", "./my_cache"],
        ):
            args = cli_module.get_args()
            assert args.cache_path == "./my_cache"


class TestCLIExecution:
    """Tests for CLI execution flow."""

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_basic_execution(self, mock_get_tasks, mock_model_class, mock_evaluator_class):
        """Test basic CLI execution flow."""
        # Setup mocks
        mock_model = Mock()
        mock_model_class.from_sentence_transformer.return_value = mock_model

        mock_tasks = [Mock(), Mock()]
        mock_get_tasks.return_value = mock_tasks

        mock_evaluator = Mock()
        mock_evaluator.run.return_value = []
        mock_evaluator_class.return_value = mock_evaluator

        # Run CLI
        with patch.object(sys, "argv", ["prog", "--model_name", "cl-nagoya/ruri-v3-30m"]):
            cli_module.main()

        # Verify model creation
        mock_model_class.from_sentence_transformer.assert_called_once()
        call_kwargs = mock_model_class.from_sentence_transformer.call_args[1]
        assert call_kwargs["model_name_or_path"] == "cl-nagoya/ruri-v3-30m"

        # Verify evaluator creation
        mock_evaluator_class.assert_called_once()
        assert mock_evaluator.run.called

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_bf16_execution(self, mock_get_tasks, mock_model_class, mock_evaluator_class):
        """Test CLI with bf16 enabled."""
        mock_model = Mock()
        mock_model_class.from_sentence_transformer.return_value = mock_model
        mock_get_tasks.return_value = [Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(sys, "argv", ["prog", "--model_name", "model", "--bf16", "true"]):
            cli_module.main()

        # Verify bf16 was passed to model
        call_kwargs = mock_model_class.from_sentence_transformer.call_args[1]
        assert "model_kwargs" in call_kwargs
        assert call_kwargs["model_kwargs"]["torch_dtype"] == torch.bfloat16

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_include_tasks(self, mock_get_tasks, mock_model_class, mock_evaluator_class):
        """Test CLI with --include argument."""
        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_get_tasks.return_value = [Mock(), Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--model_name",
                "model",
                "--include",
                '["JSTS", "JSICK"]',
            ],
        ):
            cli_module.main()

        # Verify get_jmteb_tasks was called with task_names
        mock_get_tasks.assert_called_once_with(task_names=["JSTS", "JSICK"])

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_task_types_filter(self, mock_get_tasks, mock_model_class, mock_evaluator_class):
        """Test CLI with --task_types argument."""
        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_get_tasks.return_value = [Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(
            sys,
            "argv",
            ["prog", "--model_name", "model", "--task_types", '["STS"]'],
        ):
            cli_module.main()

        # Verify get_jmteb_tasks was called with task_types
        mock_get_tasks.assert_called_once_with(task_types=["STS"])

    @patch("jmteb.v2.__main__.load_prompts")
    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_prompt_profile_loading(
        self,
        mock_get_tasks,
        mock_model_class,
        mock_evaluator_class,
        mock_load_prompts,
    ):
        """Test CLI with --prompt_profile argument."""
        mock_prompts = {"query": "query: ", "passage": "passage: "}
        mock_load_prompts.return_value = mock_prompts
        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_get_tasks.return_value = [Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--model_name",
                "model",
                "--prompt_profile",
                "prompts/e5.yaml",
            ],
        ):
            cli_module.main()

        # Verify prompts were loaded and passed to model
        mock_load_prompts.assert_called_once_with("prompts/e5.yaml")
        call_kwargs = mock_model_class.from_sentence_transformer.call_args[1]
        assert call_kwargs["prompts"] == mock_prompts

    @patch("jmteb.v2.__main__.load_batch_sizes")
    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_task_batch_sizes_loading(
        self,
        mock_get_tasks,
        mock_model_class,
        mock_evaluator_class,
        mock_load_batch_sizes,
    ):
        """Test CLI with --task_batch_sizes argument."""
        mock_batch_sizes = {"JSTS": 128, "JSICK": 128}
        mock_load_batch_sizes.return_value = mock_batch_sizes
        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_get_tasks.return_value = [Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--model_name",
                "model",
                "--task_batch_sizes",
                "batch_sizes.yaml",
            ],
        ):
            cli_module.main()

        # Verify batch sizes were loaded and passed to evaluator
        mock_load_batch_sizes.assert_called_once_with("batch_sizes.yaml")
        call_kwargs = mock_evaluator_class.call_args[1]
        assert call_kwargs["task_batch_sizes"] == mock_batch_sizes

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_save_path_configuration(self, mock_get_tasks, mock_model_class, mock_evaluator_class):
        """Test save_path creates proper directory structure."""
        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_get_tasks.return_value = [Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--model_name",
                "cl-nagoya/ruri-v3-30m",
                "--save_path",
                "my_results",
            ],
        ):
            cli_module.main()

        # Verify save_path includes model name
        call_kwargs = mock_evaluator_class.call_args[1]
        expected_path = Path("my_results") / "cl-nagoya/ruri-v3-30m"
        assert call_kwargs["save_path"] == expected_path


class TestCLIPrecisionConfiguration:
    """Tests for precision configuration."""

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_fp16_creates_model_kwargs(self, mock_get_tasks, mock_model_class, mock_evaluator_class):
        """Test that fp16 flag creates appropriate model_kwargs."""
        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_get_tasks.return_value = [Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(sys, "argv", ["prog", "--model_name", "model", "--fp16", "true"]):
            cli_module.main()

        call_kwargs = mock_model_class.from_sentence_transformer.call_args[1]
        assert call_kwargs["model_kwargs"]["torch_dtype"] == torch.float16

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_bf16_creates_model_kwargs(self, mock_get_tasks, mock_model_class, mock_evaluator_class):
        """Test that bf16 flag creates appropriate model_kwargs."""
        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_get_tasks.return_value = [Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(sys, "argv", ["prog", "--model_name", "model", "--bf16", "true"]):
            cli_module.main()

        call_kwargs = mock_model_class.from_sentence_transformer.call_args[1]
        assert call_kwargs["model_kwargs"]["torch_dtype"] == torch.bfloat16

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_no_precision_flag_no_model_kwargs(self, mock_get_tasks, mock_model_class, mock_evaluator_class):
        """Test that no precision flag means no model_kwargs."""
        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_get_tasks.return_value = [Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(sys, "argv", ["prog", "--model_name", "model"]):
            cli_module.main()

        call_kwargs = mock_model_class.from_sentence_transformer.call_args[1]
        assert call_kwargs.get("model_kwargs") is None


class TestCLITaskFiltering:
    """Tests for task filtering logic."""

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_benchmark")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    def test_exclude_filters_tasks(
        self,
        mock_get_tasks,
        mock_get_benchmark,
        mock_model_class,
        mock_evaluator_class,
    ):
        """Test that exclude argument filters out tasks."""
        mock_task1 = Mock()
        mock_task1.metadata.name = "JSTS"
        mock_task2 = Mock()
        mock_task2.metadata.name = "JSICK"
        mock_task3 = Mock()
        mock_task3.metadata.name = "JaqketRetrieval"

        mock_benchmark = Mock()
        mock_benchmark.tasks = [mock_task1, mock_task2, mock_task3]
        mock_get_benchmark.return_value = mock_benchmark

        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(
            sys,
            "argv",
            ["prog", "--model_name", "model", "--exclude", '["JSTS"]'],
        ):
            cli_module.main()

        # Verify evaluator was created with filtered tasks
        call_kwargs = mock_evaluator_class.call_args[1]
        filtered_tasks = call_kwargs["tasks"]
        assert len(filtered_tasks) == 2
        assert mock_task1 not in filtered_tasks
        assert mock_task2 in filtered_tasks
        assert mock_task3 in filtered_tasks


class TestCLIIntegration:
    """Integration tests for complete CLI workflows."""

    @patch("jmteb.v2.__main__.JMTEBV2Evaluator")
    @patch("jmteb.v2.__main__.JMTEBModel")
    @patch("jmteb.v2.__main__.get_jmteb_tasks")
    @patch("jmteb.v2.__main__.load_prompts")
    @patch("jmteb.v2.__main__.load_batch_sizes")
    def test_full_configuration(
        self,
        mock_load_batch_sizes,
        mock_load_prompts,
        mock_get_tasks,
        mock_model_class,
        mock_evaluator_class,
    ):
        """Test CLI with all configuration options."""
        mock_prompts = {"query": "query: "}
        mock_batch_sizes = {"JSTS": 128}
        mock_load_prompts.return_value = mock_prompts
        mock_load_batch_sizes.return_value = mock_batch_sizes

        mock_model_class.from_sentence_transformer.return_value = Mock()
        mock_get_tasks.return_value = [Mock()]
        mock_evaluator_class.return_value.run.return_value = []

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--model_name",
                "cl-nagoya/ruri-v3-30m",
                "--bf16",
                "true",
                "--batch_size",
                "64",
                "--prompt_profile",
                "prompts/ruri-v3.yaml",
                "--task_batch_sizes",
                "batch_sizes.yaml",
                "--save_path",
                "results",
                "--include",
                '["JSTS", "JSICK"]',
            ],
        ):
            cli_module.main()

        # Verify all components were configured correctly
        assert mock_load_prompts.called
        assert mock_load_batch_sizes.called
        assert mock_model_class.from_sentence_transformer.called
        assert mock_get_tasks.called
        assert mock_evaluator_class.called
        assert mock_evaluator_class.return_value.run.called
