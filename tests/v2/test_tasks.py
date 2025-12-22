"""
Tests for JMTEB v2.0 task utilities.
"""

from unittest.mock import Mock, patch

import pytest

from jmteb.v2 import tasks


class TestTaskUtilities:
    """Tests for task utility functions."""

    @patch("jmteb.v2.tasks.mteb.get_benchmark")
    def test_get_jmteb_benchmark(self, mock_get_benchmark):
        """Test getting JMTEB benchmark."""
        mock_benchmark = Mock()
        mock_get_benchmark.return_value = mock_benchmark

        result = tasks.get_jmteb_benchmark()

        mock_get_benchmark.assert_called_once_with("JMTEB(v2)")
        assert result == mock_benchmark

    @patch("jmteb.v2.tasks.mteb.get_benchmark")
    def test_get_jmteb_lite_benchmark(self, mock_get_benchmark):
        """Test getting JMTEB-lite benchmark."""
        mock_benchmark = Mock()
        mock_get_benchmark.return_value = mock_benchmark

        result = tasks.get_jmteb_lite_benchmark()

        mock_get_benchmark.assert_called_once_with("JMTEB-lite(v1)")
        assert result == mock_benchmark

    @patch("jmteb.v2.tasks.get_jmteb_benchmark")
    def test_get_jmteb_tasks_all(self, mock_benchmark):
        """Test getting all JMTEB tasks."""
        mock_task1 = Mock()
        mock_task1.metadata.name = "JSTS"
        mock_task2 = Mock()
        mock_task2.metadata.name = "JSICK"

        mock_benchmark.return_value.tasks = [mock_task1, mock_task2]

        result = tasks.get_jmteb_tasks()

        assert len(result) == 2
        assert result[0] == mock_task1
        assert result[1] == mock_task2

    @patch("jmteb.v2.tasks.get_jmteb_benchmark")
    def test_get_jmteb_tasks_filter_by_names(self, mock_benchmark):
        """Test filtering tasks by names."""
        mock_task1 = Mock()
        mock_task1.metadata.name = "JSTS"
        mock_task2 = Mock()
        mock_task2.metadata.name = "JSICK"
        mock_task3 = Mock()
        mock_task3.metadata.name = "JaqketRetrieval"

        mock_benchmark.return_value.tasks = [mock_task1, mock_task2, mock_task3]

        result = tasks.get_jmteb_tasks(task_names=["JSTS", "JSICK"])

        assert len(result) == 2
        assert mock_task1 in result
        assert mock_task2 in result
        assert mock_task3 not in result

    @patch("jmteb.v2.tasks.get_jmteb_benchmark")
    def test_get_jmteb_tasks_filter_by_type(self, mock_benchmark):
        """Test filtering tasks by type."""
        mock_task1 = Mock()
        mock_task1.metadata.name = "JSTS"
        mock_task1.metadata.type = "STS"
        mock_task2 = Mock()
        mock_task2.metadata.name = "JaqketRetrieval"
        mock_task2.metadata.type = "Retrieval"

        mock_benchmark.return_value.tasks = [mock_task1, mock_task2]

        result = tasks.get_jmteb_tasks(task_types=["STS"])

        assert len(result) == 1
        assert result[0] == mock_task1

    @patch("jmteb.v2.tasks.get_jmteb_benchmark")
    def test_get_jmteb_tasks_filter_by_language(self, mock_benchmark):
        """Test filtering tasks by language."""
        mock_task1 = Mock()
        mock_task1.metadata.name = "JSTS"
        mock_task1.metadata.languages = ["jpn"]
        mock_task2 = Mock()
        mock_task2.metadata.name = "SomeTask"
        mock_task2.metadata.languages = ["eng"]

        mock_benchmark.return_value.tasks = [mock_task1, mock_task2]

        result = tasks.get_jmteb_tasks(languages=["jpn"])

        assert len(result) == 1
        assert result[0] == mock_task1

    @patch("jmteb.v2.tasks.get_jmteb_tasks")
    def test_get_task_by_name_success(self, mock_get_tasks):
        """Test getting a task by name successfully."""
        mock_task = Mock()
        mock_task.metadata.name = "JSTS"
        mock_get_tasks.return_value = [mock_task]

        result = tasks.get_task_by_name("JSTS")

        assert result == mock_task
        mock_get_tasks.assert_called_once_with(task_names=["JSTS"])

    @patch("jmteb.v2.tasks.get_jmteb_tasks")
    def test_get_task_by_name_not_found(self, mock_get_tasks):
        """Test getting a task by name when not found."""
        mock_get_tasks.return_value = []

        with pytest.raises(ValueError, match="Task 'InvalidTask' not found"):
            tasks.get_task_by_name("InvalidTask")

    def test_get_task_category(self):
        """Test getting task category."""
        assert tasks.get_task_category("JSTS") == "STS"
        assert tasks.get_task_category("JaqketRetrieval") == "Retrieval"
        assert tasks.get_task_category("AmazonReviewsClassification") == "Classification"
        assert tasks.get_task_category("LivedoorNewsClustering.v2") == "Clustering"
        assert tasks.get_task_category("ESCIReranking") == "Reranking"
        assert tasks.get_task_category("UnknownTask") == "Unknown"

    def test_convert_v1_task_name(self):
        """Test converting v1 task names to v2."""
        assert tasks.convert_v1_task_name("jsts") == "JSTS"
        assert tasks.convert_v1_task_name("jaqket") == "JaqketRetrieval"
        assert tasks.convert_v1_task_name("livedoor_news") == "LivedoorNewsClustering.v2"
        # Unknown task should return as-is
        assert tasks.convert_v1_task_name("unknown_task") == "unknown_task"

    def test_jmteb_tasks_constant(self):
        """Test that JMTEB_TASKS constant is properly defined."""
        assert len(tasks.JMTEB_TASKS) == 28
        assert "JSTS" in tasks.JMTEB_TASKS
        assert "JSICK" in tasks.JMTEB_TASKS
        assert "JaqketRetrieval" in tasks.JMTEB_TASKS

    def test_task_categories_constant(self):
        """Test that TASK_CATEGORIES constant covers all tasks."""
        for task_name in tasks.JMTEB_TASKS:
            assert task_name in tasks.TASK_CATEGORIES
