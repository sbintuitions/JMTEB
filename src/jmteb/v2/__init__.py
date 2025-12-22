"""
JMTEB v2.0 - MTEB Integration Layer

This module provides the v2.0 architecture that integrates MTEB as the underlying
evaluation engine while maintaining backward compatibility with JMTEB v1.x APIs.
"""

from jmteb.v2.adapters import JMTEBModel
from jmteb.v2.evaluator import JMTEBV2Evaluator
from jmteb.v2.tasks import (
    JMTEB_TASKS,
    JMTEB_LITE_TASKS,
    get_jmteb_tasks,
    get_jmteb_lite_tasks,
    get_jmteb_benchmark,
    get_jmteb_lite_benchmark,
    get_task_by_name,
)
from jmteb.v2.utils import (
    load_prompts,
    load_batch_sizes,
    save_results,
    load_summary,
    save_summary,
)

__all__ = [
    "JMTEBModel",
    "JMTEBV2Evaluator",
    "JMTEB_TASKS",
    "JMTEB_LITE_TASKS",
    "get_jmteb_tasks",
    "get_jmteb_lite_tasks",
    "get_jmteb_benchmark",
    "get_jmteb_lite_benchmark",
    "get_task_by_name",
    "load_prompts",
    "load_batch_sizes",
    "save_results",
    "load_summary",
    "save_summary",
]
