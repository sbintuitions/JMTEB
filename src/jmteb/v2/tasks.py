"""
JMTEB v2.0 task definitions and utilities using MTEB framework.
"""

from __future__ import annotations

import mteb
from mteb import AbsTask


# JMTEB v2.0 consists of 28 tasks aligned with MTEB's JMTEB(v2) benchmark
JMTEB_TASKS = [
    # Clustering (3 tasks)
    "LivedoorNewsClustering.v2",
    "MewsC16JaClustering",
    "SIB200ClusteringS2S",
    # Classification (7 tasks)
    "AmazonReviewsClassification",
    "AmazonCounterfactualClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "JapaneseSentimentClassification",
    "SIB200Classification",
    "WRIMEClassification",
    # STS (2 tasks)
    "JSTS",
    "JSICK",
    # Retrieval (11 tasks)
    "JaqketRetrieval",
    "MrTidyRetrieval",
    "JaGovFaqsRetrieval",
    "NLPJournalTitleAbsRetrieval.V2",
    "NLPJournalTitleIntroRetrieval.V2",
    "NLPJournalAbsIntroRetrieval.V2",
    "NLPJournalAbsArticleRetrieval.V2",
    "JaCWIRRetrieval",
    "MIRACLRetrieval",
    "MintakaRetrieval",
    "MultiLongDocRetrieval",
    # Reranking (5 tasks)
    "ESCIReranking",
    "JQaRAReranking",
    "JaCWIRReranking",
    "MIRACLReranking",
    "MultiLongDocReranking",
]


# JMTEB-lite consists of the same 28 tasks as JMTEB but with reduced corpus sizes
# for faster evaluation (~5x speedup with 0.97 Spearman correlation to full JMTEB)
# The lightweight tasks (with "Lite" suffix) have reduced corpus sizes:
# JaqketRetrievalLite, MrTyDiJaRetrievalLite, JaCWIRRetrievalLite,
# MIRACLJaRetrievalLite, JQaRARerankingLite, JaCWIRRerankingLite
JMTEB_LITE_TASKS = [
    # Clustering (3 tasks)
    "LivedoorNewsClustering.v2",
    "MewsC16JaClustering",
    "SIB200ClusteringS2S",
    # Classification (7 tasks)
    "AmazonReviewsClassification",
    "AmazonCounterfactualClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "JapaneseSentimentClassification",
    "SIB200Classification",
    "WRIMEClassification",
    # STS (2 tasks)
    "JSTS",
    "JSICK",
    # Retrieval (11 tasks)
    "JaqketRetrievalLite",
    "MrTyDiJaRetrievalLite",
    "JaGovFaqsRetrieval",
    "NLPJournalTitleAbsRetrieval.V2",
    "NLPJournalTitleIntroRetrieval.V2",
    "NLPJournalAbsIntroRetrieval.V2",
    "NLPJournalAbsArticleRetrieval.V2",
    "JaCWIRRetrievalLite",
    "MIRACLJaRetrievalLite",
    "MintakaRetrieval",
    "MultiLongDocRetrieval",
    # Reranking (5 tasks)
    "ESCIReranking",
    "JQaRARerankingLite",
    "JaCWIRRerankingLite",
    "MIRACLReranking",
    "MultiLongDocReranking",
]


# Task type categorization for summary generation
TASK_CATEGORIES = {
    # Classification
    "AmazonReviewsClassification": "Classification",
    "AmazonCounterfactualClassification": "Classification",
    "MassiveIntentClassification": "Classification",
    "MassiveScenarioClassification": "Classification",
    "JapaneseSentimentClassification": "Classification",
    "SIB200Classification": "Classification",
    "WRIMEClassification": "Classification",
    # Clustering
    "LivedoorNewsClustering.v2": "Clustering",
    "MewsC16JaClustering": "Clustering",
    "SIB200ClusteringS2S": "Clustering",
    # STS
    "JSTS": "STS",
    "JSICK": "STS",
    # Retrieval
    "JaqketRetrieval": "Retrieval",
    "MrTidyRetrieval": "Retrieval",
    "JaGovFaqsRetrieval": "Retrieval",
    "NLPJournalTitleAbsRetrieval.V2": "Retrieval",
    "NLPJournalTitleIntroRetrieval.V2": "Retrieval",
    "NLPJournalAbsIntroRetrieval.V2": "Retrieval",
    "NLPJournalAbsArticleRetrieval.V2": "Retrieval",
    "JaCWIRRetrieval": "Retrieval",
    "MIRACLRetrieval": "Retrieval",
    "MintakaRetrieval": "Retrieval",
    "MultiLongDocRetrieval": "Retrieval",
    # Reranking
    "ESCIReranking": "Reranking",
    "JQaRAReranking": "Reranking",
    "JaCWIRReranking": "Reranking",
    "MIRACLReranking": "Reranking",
    "MultiLongDocReranking": "Reranking",
}


# Mapping of JMTEB v1 task names to MTEB task names (for backward compatibility)
V1_TO_V2_TASK_MAPPING = {
    # v1 name -> v2 name
    "livedoor_news": "LivedoorNewsClustering.v2",
    "mewsc16": "MewsC16JaClustering",
    "amazon_review_classification": "AmazonReviewsClassification",
    "amazon_counterfactual_classification": "AmazonCounterfactualClassification",
    "massive_intent_classification": "MassiveIntentClassification",
    "massive_scenario_classification": "MassiveScenarioClassification",
    "jsts": "JSTS",
    "jsick": "JSICK",
    "jaqket": "JaqketRetrieval",
    "mrtydi": "MrTidyRetrieval",
    "jagovfaqs_22k": "JaGovFaqsRetrieval",
    "nlp_journal_title_abs": "NLPJournalTitleAbsRetrieval.V2",
    "nlp_journal_title_intro": "NLPJournalTitleIntroRetrieval.V2",
    "nlp_journal_abs_intro": "NLPJournalAbsIntroRetrieval.V2",
    "nlp_journal_abs_article": "NLPJournalAbsArticleRetrieval.V2",
    "jacwir_retrieval": "JaCWIRRetrieval",
    "miracl_retrieval": "MIRACLRetrieval",
    "esci": "ESCIReranking",
    "jqara": "JQaRAReranking",
    "jacwir_reranking": "JaCWIRReranking",
    "miracl_reranking": "MIRACLReranking",
}


def get_jmteb_benchmark() -> mteb.Benchmark:
    """
    Get the JMTEB(v2) benchmark from MTEB.

    Returns:
        MTEB Benchmark object containing all JMTEB tasks

    Example:
        >>> benchmark = get_jmteb_benchmark()
        >>> print(f"JMTEB contains {len(benchmark.tasks)} tasks")
        >>> print(benchmark.tasks[0].metadata.name)
    """
    return mteb.get_benchmark("JMTEB(v2)")


def get_jmteb_lite_benchmark() -> mteb.Benchmark:
    """
    Get the JMTEB-lite benchmark from MTEB.

    JMTEB-lite is a lightweight version with reduced corpus sizes for
    faster evaluation (~5x faster) while maintaining high correlation
    with full JMTEB results.

    Returns:
        MTEB Benchmark object containing all JMTEB-lite tasks

    Example:
        >>> benchmark = get_jmteb_lite_benchmark()
        >>> print(f"JMTEB-lite contains {len(benchmark.tasks)} tasks")
    """
    return mteb.get_benchmark("JMTEB-lite(v1)")


def _get_tasks_from_benchmark(
    benchmark: mteb.Benchmark,
    task_names: list[str] | None = None,
    task_types: list[str] | None = None,
) -> list[AbsTask]:
    """
    Internal helper to get tasks from a benchmark with optional filtering.

    Args:
        benchmark: MTEB Benchmark object
        task_names: List of specific task names to retrieve. If None, returns all tasks.
        task_types: Filter tasks by type (e.g., ["Retrieval", "Classification"]).

    Returns:
        List of MTEB task objects
    """
    tasks = benchmark.tasks

    # Filter by task names if specified
    if task_names is not None:
        tasks = [task for task in tasks if task.metadata.name in task_names]

    # Filter by task types if specified
    if task_types is not None:
        tasks = [task for task in tasks if task.metadata.type in task_types]

    # Extract task names and use mteb.get_tasks to restrict all tasks to Japanese only
    # This properly handles multilingual tasks by restricting them to jpn subset
    task_name_list = [task.metadata.name for task in tasks]
    tasks = mteb.get_tasks(tasks=task_name_list, languages=["jpn"])

    return tasks


def get_jmteb_tasks(
    task_names: list[str] | None = None,
    task_types: list[str] | None = None,
) -> list[AbsTask]:
    """
    Get JMTEB tasks with optional filtering.

    Args:
        task_names: List of specific task names to retrieve. If None, returns all tasks.
        task_types: Filter tasks by type (e.g., ["Retrieval", "Classification"]).

    Returns:
        List of MTEB task objects

    Example:
        >>> # Get all JMTEB tasks
        >>> tasks = get_jmteb_tasks()
        >>>
        >>> # Get specific tasks
        >>> tasks = get_jmteb_tasks(task_names=["JSTS", "JSICK"])
        >>>
        >>> # Get all retrieval tasks
        >>> tasks = get_jmteb_tasks(task_types=["Retrieval"])
    """
    return _get_tasks_from_benchmark(
        get_jmteb_benchmark(), task_names=task_names, task_types=task_types
    )


def get_jmteb_lite_tasks(
    task_names: list[str] | None = None,
    task_types: list[str] | None = None,
) -> list[AbsTask]:
    """
    Get JMTEB-lite tasks with optional filtering.

    JMTEB-lite provides ~5x faster evaluation with reduced corpus sizes
    while maintaining high correlation (0.97 Spearman) with full JMTEB results.

    Args:
        task_names: List of specific task names to retrieve. If None, returns all tasks.
        task_types: Filter tasks by type (e.g., ["Retrieval", "Classification"]).

    Returns:
        List of MTEB task objects

    Example:
        >>> # Get all JMTEB-lite tasks
        >>> tasks = get_jmteb_lite_tasks()
        >>>
        >>> # Get specific tasks
        >>> tasks = get_jmteb_lite_tasks(task_names=["JSTS", "JSICK"])
        >>>
        >>> # Get all retrieval tasks
        >>> tasks = get_jmteb_lite_tasks(task_types=["Retrieval"])
    """
    return _get_tasks_from_benchmark(
        get_jmteb_lite_benchmark(), task_names=task_names, task_types=task_types
    )


def get_task_by_name(task_name: str, lite: bool = False) -> AbsTask:
    """
    Get a single task by name.

    Args:
        task_name: Name of the task (MTEB format)
        lite: If True, search in JMTEB-lite benchmark; otherwise search in JMTEB

    Returns:
        MTEB task object

    Raises:
        ValueError: If task name is not found

    Example:
        >>> # Get from JMTEB
        >>> task = get_task_by_name("JSTS")
        >>> print(task.metadata.description)
        >>>
        >>> # Get from JMTEB-lite
        >>> task = get_task_by_name("JaqketRetrievalLite", lite=True)
    """
    if lite:
        tasks = get_jmteb_lite_tasks(task_names=[task_name])
        available_tasks = JMTEB_LITE_TASKS
        benchmark_name = "JMTEB-lite"
    else:
        tasks = get_jmteb_tasks(task_names=[task_name])
        available_tasks = JMTEB_TASKS
        benchmark_name = "JMTEB"

    if not tasks:
        raise ValueError(
            f"Task '{task_name}' not found in {benchmark_name} benchmark. "
            f"Available tasks: {available_tasks}"
        )
    return tasks[0]


def get_task_category(task_name: str) -> str:
    """
    Get the category/type of a task.

    Args:
        task_name: Name of the task

    Returns:
        Task category (e.g., "Classification", "Retrieval", etc.)

    Example:
        >>> category = get_task_category("JSTS")
        >>> print(category)  # "STS"
    """
    return TASK_CATEGORIES.get(task_name, "Unknown")


def convert_v1_task_name(v1_name: str) -> str:
    """
    Convert JMTEB v1 task name to v2 (MTEB) format.

    Args:
        v1_name: JMTEB v1 task name

    Returns:
        MTEB task name

    Example:
        >>> v2_name = convert_v1_task_name("jsts")
        >>> print(v2_name)  # "JSTS"
    """
    return V1_TO_V2_TASK_MAPPING.get(v1_name, v1_name)
