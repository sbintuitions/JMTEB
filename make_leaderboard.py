import json
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

dataset_name_aliases = {
    "amazon_counterfactual_classification": "amazon_counterfactual",
    "amazon_review_classification": "amazon_review",
    "massive_intent_classification": "massive_intent",
    "massive_scenario_classification": "massive_scenario",
}

TASK_ORDER = ["Retrieval", "STS", "Classification", "Reranking", "Clustering", "PairClassification"]
SUMMARY_KEY = "Summary"

"""
Collects the results from the results folder.
"""
# {task_name: {model_signature: {(dataset_name, metric_name): score}}}
all_results: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
for summary_file in Path("docs/results").rglob("summary.json"):
    if not summary_file.exists():
        continue

    with open(summary_file) as f:
        summary = json.load(f)

    org_name = summary_file.parent.parent.name
    model_name = summary_file.parent.name
    model_signature = f"{org_name}/{model_name}"

    for task_name, task_results in summary.items():
        task_results_formatted: dict[str, float] = {}
        task_scores: list[float] = []
        for dataset_name, metric_dict in task_results.items():
            metric_name, score = next(iter(metric_dict.items()))
            dataset_name = dataset_name_aliases.get(dataset_name, dataset_name)
            task_results_formatted[f"{dataset_name}<br>({metric_name})"] = score
            task_scores.append(score)
        all_results[task_name][model_signature] = task_results_formatted
        all_results[SUMMARY_KEY][model_signature][task_name] = sum(task_scores) / len(task_scores)

"""
Creates markdown tables for each task.
"""


def format_score(score: float) -> str:
    return f"{score * 100:.2f}"


AVG_COLUMN_NAME = "Avg."
markdown_tables: dict[str, str] = {}
for task_name, task_results in all_results.items():
    # format to markdown table
    dataset_keys = list(task_results[next(iter(task_results))].keys())
    if task_name == SUMMARY_KEY:
        dataset_keys = TASK_ORDER

    header = ["Model", AVG_COLUMN_NAME, *dataset_keys]
    table_list: list[list[str | float]] = []
    for model_signature, dataset_scores in task_results.items():
        model_scores = [dataset_scores[k] for k in dataset_keys]
        average_score = sum(model_scores) / len(model_scores)
        table_list.append([model_signature, average_score, *model_scores])

    # sort by the average score
    avg_idx = header.index(AVG_COLUMN_NAME)
    table_list.sort(key=lambda x: x[avg_idx], reverse=True)

    # make the highest score in each dataset bold
    for dataset_name in [AVG_COLUMN_NAME, *dataset_keys]:
        task_idx = header.index(dataset_name)
        max_score = max(row[task_idx] for row in table_list)
        for row in table_list:
            if row[task_idx] == max_score:
                row[task_idx] = f"**{format_score(row[task_idx])}**"
            else:
                row[task_idx] = format_score(row[task_idx])

    # add header
    table_list.insert(0, ["Model", AVG_COLUMN_NAME, *dataset_keys])
    markdown_table = tabulate(table_list, headers="firstrow", tablefmt="pipe")
    markdown_tables[task_name] = markdown_table

"""
Dump the markdown tables to a file.
"""
with open("leaderboard.md", "w") as f:
    f.write("# Leaderboard\n")
    f.write(
        "This leaderboard shows the results stored under `docs/results`. The scores are all multiplied by 100.\n\n"
    )
    for task_name in [SUMMARY_KEY, *TASK_ORDER]:
        markdown_table = markdown_tables[task_name]
        f.write(f"## {task_name}\n")

        if task_name == SUMMARY_KEY:
            f.write("\nThe summary shows the average scores within each task.\n\n")

        f.write(markdown_table)
        f.write("\n\n")
