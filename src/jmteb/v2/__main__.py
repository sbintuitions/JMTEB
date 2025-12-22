"""
JMTEB v2.0 main entry point.

This module provides the CLI interface for running JMTEB v2.0 evaluations using MTEB.
"""

from __future__ import annotations

import torch
from jsonargparse import ArgumentParser
from loguru import logger

from jmteb.v2.adapters import JMTEBModel
from jmteb.v2.evaluator import JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_benchmark, get_jmteb_tasks
from jmteb.v2.utils import load_batch_sizes, load_prompts


def get_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="JMTEB v2.0 - Japanese Massive Text Embedding Benchmark")

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the model to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Default batch size for encoding",
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        default=False,
        help="Use FP16 precision",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=False,
        help="Use BF16 precision",
    )

    # Task selection
    parser.add_argument(
        "--include",
        type=list[str],
        default=None,
        help="List of task names to include in evaluation",
    )
    parser.add_argument(
        "--exclude",
        type=list[str],
        default=None,
        help="List of task names to exclude from evaluation",
    )
    parser.add_argument(
        "--task_types",
        type=list[str],
        default=None,
        help="List of task types to evaluate (e.g., ['Retrieval', 'Classification'])",
    )

    # Configuration files
    parser.add_argument(
        "--prompt_profile",
        type=str,
        default=None,
        help="Path to prompt configuration YAML file",
    )
    parser.add_argument(
        "--task_batch_sizes",
        type=str,
        default=None,
        help="Path to YAML file with per-task batch sizes",
    )

    # Output configuration
    parser.add_argument(
        "--save_path",
        type=str,
        default="results_v2",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite cached results and reevaluate",
    )
    parser.add_argument(
        "--generate_summary",
        type=bool,
        default=True,
        help="Generate/update summary.json file",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="./cached_results",
        help="Path for caching intermediate results",
    )

    return parser.parse_args()


def main():
    """Main function to run JMTEB v2.0 evaluation."""
    args = get_args()

    # Prepare model_kwargs based on fp16/bf16
    model_kwargs = {}
    if args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
        logger.info("Using FP16 precision")
    elif args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
        logger.info("Using BF16 precision")

    # Load prompts if provided
    prompts = None
    if args.prompt_profile:
        prompts = load_prompts(args.prompt_profile)
        logger.info(f"Loaded prompts from {args.prompt_profile}")
        logger.info(f"Prompt keys: {list(prompts.keys())}")

    # Create model
    logger.info(f"Loading model: {args.model_name}")
    model = JMTEBModel.from_sentence_transformer(
        model_name_or_path=args.model_name,
        model_kwargs=model_kwargs if model_kwargs else None,
        prompts=prompts,
    )
    logger.info("Model loaded successfully")

    # Get tasks
    if args.include:
        logger.info(f"Including specific tasks: {args.include}")
        tasks = get_jmteb_tasks(task_names=args.include)
    elif args.exclude:
        logger.info(f"Excluding tasks: {args.exclude}")
        benchmark = get_jmteb_benchmark()
        tasks = [t for t in benchmark.tasks if t.metadata.name not in args.exclude]
    elif args.task_types:
        logger.info(f"Filtering by task types: {args.task_types}")
        tasks = get_jmteb_tasks(task_types=args.task_types)
    else:
        logger.info("Evaluating all JMTEB tasks")
        tasks = get_jmteb_tasks()

    logger.info(f"Loaded {len(tasks)} tasks")

    # Load task-specific batch sizes if provided
    task_batch_sizes = {}
    if args.task_batch_sizes:
        task_batch_sizes = load_batch_sizes(args.task_batch_sizes)
        logger.info(f"Loaded task-specific batch sizes from {args.task_batch_sizes}")

    # Create save path for model
    from pathlib import Path

    model_save_path = Path(args.save_path) / args.model_name
    logger.info(f"Results will be saved to: {model_save_path}")

    # Create evaluator
    evaluator = JMTEBV2Evaluator(
        model=model,
        tasks=tasks,
        save_path=model_save_path,
        batch_size=args.batch_size,
        task_batch_sizes=task_batch_sizes,
        overwrite_cache=args.overwrite_cache,
        generate_summary=args.generate_summary,
        cache_path=args.cache_path,
    )

    # Run evaluation
    logger.info("=" * 80)
    logger.info("Starting evaluation")
    logger.info("=" * 80)

    results = evaluator.run()

    logger.info("=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)

    if results:
        logger.info(f"Results available in: {model_save_path}")


if __name__ == "__main__":
    main()
