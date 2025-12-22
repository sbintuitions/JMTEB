"""
JMTEB v2.0 utility functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from jmteb.v2.tasks import TASK_CATEGORIES


def load_prompts(prompt_config: str | Path) -> dict[str, str]:
    """
    Load prompt configuration from a YAML file.

    Args:
        prompt_config: Path to YAML file containing prompts

    Returns:
        Dictionary mapping task types to prompt templates

    Example:
        >>> prompts = load_prompts("prompts/e5.yaml")
        >>> print(prompts.get("query", ""))
    """
    prompt_path = Path(prompt_config) if isinstance(prompt_config, str) else prompt_config

    with open(prompt_path, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    return prompts if prompts is not None else {}


def load_batch_sizes(batch_size_config: str | Path) -> dict[str, int]:
    """
    Load per-task batch size configuration from a YAML file.

    Args:
        batch_size_config: Path to YAML file containing batch sizes

    Returns:
        Dictionary mapping task names to batch sizes

    Example:
        >>> batch_sizes = load_batch_sizes("batch_sizes.yaml")
        >>> print(batch_sizes.get("JSTS", 32))
    """
    batch_size_path = Path(batch_size_config) if isinstance(batch_size_config, str) else batch_size_config

    with open(batch_size_path, encoding="utf-8") as f:
        batch_sizes = yaml.safe_load(f)

    return batch_sizes if batch_sizes is not None else {}


def load_summary(save_path: str | Path) -> dict:
    """
    Load existing summary.json if it exists.

    Args:
        save_path: Directory containing summary.json

    Returns:
        Dictionary containing summary data, or empty dict if not found
    """
    summary_path = Path(save_path) / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def save_summary(summary: dict, save_path: str | Path):
    """
    Save summary.json to disk.

    Args:
        summary: Summary dictionary to save
        save_path: Directory to save summary.json in
    """
    summary_path = Path(save_path) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Error saving summary: {e}")


def extract_and_update_summary(
    task_name: str,
    main_metric: str,
    save_path: str | Path,
    summary: dict,
    eval_time: float = -1,
):
    """
    Extract main score from task result and update summary with timing.

    Args:
        task_name: Name of the task
        main_metric: Main metric for the task
        save_path: Path where results are saved
        summary: Summary dictionary to update
        eval_time: Time taken to evaluate in seconds, or -1 if cached
    """
    task_category = TASK_CATEGORIES.get(task_name)
    if not task_category:
        return

    result_path = Path(save_path) / f"{task_name}.json"

    if not result_path.exists():
        return

    with open(result_path) as f:
        result_data = json.load(f)

    # Determine the split to use
    # JSTS uses validation split, MLDR tasks use dev split, others use test
    if task_name == "JSTS":
        split = "validation"
    elif task_name.startswith("MultiLongDoc"):
        split = "dev"
    else:
        split = "test"

    # Extract score from result
    if split in result_data and len(result_data[split]) > 0:
        score = result_data[split][0].get("main_score")

        if score is not None:
            # Create task key (convert MTEB name back to simpler format for summary)
            task_key = _get_task_key(task_name)

            # Update summary
            if task_category not in summary:
                summary[task_category] = {}

            summary[task_category][task_key] = {
                "main_metric": main_metric,
                "main_score": score * 100,  # Convert to percentage
                "eval_time (s)": "%.2f" % eval_time if eval_time >= 0 else "cached",
            }

            # Save immediately
            save_summary(summary, save_path)


def _get_task_key(task_name: str) -> str:
    """
    Convert MTEB task name to a simpler key for summary.

    Args:
        task_name: MTEB task name

    Returns:
        Simplified task key
    """
    # Map common task names to simpler keys
    task_key_mapping = {
        "LivedoorNewsClustering.v2": "livedoor_news",
        "MewsC16JaClustering": "mewsc16",
        "SIB200ClusteringS2S": "sib200_japanese_clustering",
        "AmazonReviewsClassification": "amazon_review_classification",
        "AmazonCounterfactualClassification": "amazon_counterfactual_classification",
        "MassiveIntentClassification": "massive_intent_classification",
        "MassiveScenarioClassification": "massive_scenario_classification",
        "JapaneseSentimentClassification": "japanese_sentiment_classification",
        "SIB200Classification": "sib200_japanese_classification",
        "WRIMEClassification": "wrime_classification",
        "JSTS": "jsts",
        "JSICK": "jsick",
        "JaqketRetrieval": "jaqket",
        "MrTidyRetrieval": "mrtydi",
        "JaGovFaqsRetrieval": "jagovfaqs_22k",
        "NLPJournalTitleAbsRetrieval.V2": "nlp_journal_title_abs",
        "NLPJournalTitleIntroRetrieval.V2": "nlp_journal_title_intro",
        "NLPJournalAbsIntroRetrieval.V2": "nlp_journal_abs_intro",
        "NLPJournalAbsArticleRetrieval.V2": "nlp_journal_abs_article",
        "JaCWIRRetrieval": "jacwir_retrieval",
        "MIRACLRetrieval": "miracl_retrieval",
        "MintakaRetrieval": "mintaka_retrieval",
        "MultiLongDocRetrieval": "mldr_retrieval",
        "ESCIReranking": "esci",
        "JQaRAReranking": "jqara",
        "JaCWIRReranking": "jacwir_reranking",
        "MIRACLReranking": "miracl_reranking",
        "MultiLongDocReranking": "mldr_reranking",
    }

    return task_key_mapping.get(task_name, task_name.lower())


def save_results(
    results: dict,
    save_path: str | Path,
    filename: str = "results.json",
):
    """
    Save evaluation results to a JSON file.

    Args:
        results: Results dictionary to save
        save_path: Directory to save results in
        filename: Name of the file to save
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    result_file = save_path / filename
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
