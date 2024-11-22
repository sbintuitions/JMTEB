from __future__ import annotations

import json
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser
from loguru import logger

from jmteb.embedders import TextEmbedder
from jmteb.evaluators import EmbeddingEvaluator
from jmteb.utils.dist import is_main_process
from jmteb.utils.score_recorder import JsonScoreRecorder


def main(
    text_embedder: TextEmbedder,
    evaluators: dict[str, EmbeddingEvaluator],
    save_dir: str | None = None,
    overwrite_cache: bool = False,
):
    if is_main_process():
        logger.info(f"Start evaluating the following tasks\n{list(evaluators.keys())}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    score_recorder = JsonScoreRecorder(save_dir)

    for eval_name, evaluator in evaluators.items():
        if is_main_process():
            logger.info(f"Evaluating {eval_name}")

        cache_dir = None
        if save_dir is not None:
            cache_dir = Path(save_dir) / "cache" / eval_name

        metrics = evaluator(text_embedder, cache_dir=cache_dir, overwrite_cache=overwrite_cache)
        if metrics is not None:
            score_recorder.record_task_scores(
                scores=metrics,
                dataset_name=eval_name,
                task_name=evaluator.__class__.__name__.replace("Evaluator", ""),
            )
            if getattr(evaluator, "log_predictions", False):
                score_recorder.record_predictions(
                    metrics, eval_name, evaluator.__class__.__name__.replace("Evaluator", "")
                )

            logger.info(f"Results for {eval_name}\n{json.dumps(metrics.as_dict(), indent=4, ensure_ascii=False)}")

    if save_dir and is_main_process():
        logger.info(f"Saving result summary to {Path(save_dir) / 'summary.json'}")
        score_recorder.record_summary()


if __name__ == "__main__":
    parser = ArgumentParser(parser_mode="jsonnet")

    parser.add_subclass_arguments(TextEmbedder, nested_key="embedder", required=True)
    parser.add_argument(
        "--evaluators",
        type=dict[str, EmbeddingEvaluator],
        enable_path=True,
        default=str(Path(__file__).parent / "configs" / "jmteb.jsonnet"),
    )
    parser.add_argument("--config", action=ActionConfigFile, help="Path to the config file.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the outputs")
    parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the save_dir if it exists")
    parser.add_argument("--eval_include", type=list[str], default=None, help="Evaluators to include.")
    parser.add_argument("--eval_exclude", type=list[str], default=None, help="Evaluators to exclude.")
    parser.add_argument(
        "--log_predictions", type=bool, default=False, help="Whether to log predictions for all evaulators."
    )

    args = parser.parse_args()

    if args.eval_include is not None:
        # check if the specified evaluators are valid
        evaluator_keys = list(args.evaluators.keys())
        for include_key in args.eval_include:
            if include_key not in evaluator_keys:
                raise ValueError(f"Invalid evaluator name: {include_key}")

        # remove evaluators not in eval_include
        for key in evaluator_keys:
            if key not in args.eval_include:
                args.evaluators.pop(key)

    if args.eval_exclude is not None:
        # check if the specified evaluators are valid
        evaluator_keys = list(args.evaluators.keys())
        for exclude_key in args.eval_exclude:
            if exclude_key not in evaluator_keys:
                raise ValueError(f"Invalid evaluator name: {exclude_key}")

        # remove evaluators in eval_exclude
        for key in evaluator_keys:
            if key in args.eval_exclude:
                args.evaluators.pop(key)

    if len(args.evaluators) == 0:
        raise ValueError("No evaluator is selected. Please check the config file or the command line arguments.")

    # save config as yaml
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        parser.save(
            args,
            Path(args.save_dir) / "jmteb_config.yaml",
            format="yaml",
            overwrite=True,
            multifile=False,
            skip_check=True,
        )

    args = parser.instantiate_classes(args)
    if isinstance(args.evaluators, str):
        raise ValueError(
            "Evaluators should be a dictionary, not a string.\n"
            "Perhaps you provided a path to a config file, "
            "but the path does not exist or the config format is broken.\n"
            f"Please check {args.evaluators}"
        )

    if args.log_predictions:
        for k, v in args.evaluators.items():
            if hasattr(v, "log_predictions"):
                args.evaluators[k].log_predictions = True

    main(
        text_embedder=args.embedder,
        evaluators=args.evaluators,
        save_dir=args.save_dir,
        overwrite_cache=args.overwrite_cache,
    )
