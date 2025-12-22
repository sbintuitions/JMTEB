"""
Pytest configuration and fixtures for JMTEB v2.0 tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    model = Mock()
    model.encode = Mock(return_value=np.random.rand(10, 768))
    return model


@pytest.fixture
def mock_embedder():
    """Mock JMTEB v1 TextEmbedder."""
    embedder = Mock()
    embedder.encode = Mock(return_value=np.random.rand(10, 768).tolist())
    return embedder


@pytest.fixture
def sample_sentences():
    """Sample sentences for testing."""
    return [
        "これはテストです。",
        "日本語のテキスト埋め込み",
        "JMTEB v2.0のテスト",
        "機械学習モデルの評価",
        "自然言語処理",
    ]


@pytest.fixture
def mock_mteb_task():
    """Mock MTEB task."""
    task = Mock()
    task.metadata = Mock()
    task.metadata.name = "JSTS"
    task.metadata.main_score = "cosine_spearman"
    task.metadata.type = "STS"
    task.metadata.languages = ["jpn"]
    return task


@pytest.fixture
def sample_task_results():
    """Sample task evaluation results."""
    return {
        "validation": [
            {
                "main_score": 0.8234,
                "cosine_spearman": 0.8234,
                "cosine_pearson": 0.8156,
            }
        ]
    }
