"""
JMTEB v2.0 evaluator using MTEB framework.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import mteb
from loguru import logger
from mteb import AbsTask
from mteb.cache import ResultCache

from jmteb.v2.adapters import JMTEBModel
from jmteb.v2.utils import _get_task_key, load_summary, save_summary

if TYPE_CHECKING:
    from mteb.results import ModelResult, TaskResult

# Split used for the summary score when a task is evaluated on several splits (default: "test")
SUMMARY_SPLITS = {
    "JSTS": "validation",
    "MultiLongDocRetrieval": "dev",
}


class JMTEBV2Evaluator:
    """
    JMTEB v2.0 evaluator that uses MTEB as the underlying evaluation engine.

    This evaluator provides a high-level interface for running JMTEB benchmarks
    while leveraging MTEB's robust evaluation framework and caching.

    Example:
        >>> from jmteb.v2 import JMTEBV2Evaluator, JMTEBModel
        >>> from jmteb.v2.tasks import get_jmteb_tasks
        >>>
        >>> # Create model
        >>> model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-base")
        >>>
        >>> # Create evaluator
        >>> evaluator = JMTEBV2Evaluator(
        ...     model=model,
        ...     tasks=get_jmteb_tasks(task_names=["JSTS", "JSICK"]),
        ...     save_path="results/ruri-base"
        ... )
        >>>
        >>> # Run evaluation
        >>> results = evaluator.run()
    """

    def __init__(
        self,
        model: JMTEBModel,
        tasks: Iterable[AbsTask] | AbsTask,
        save_path: str | Path | None = None,
        batch_size: int = 32,
        task_batch_sizes: dict[str, int] | None = None,
        cache_path: str | Path | None = None,
        overwrite_cache: bool = False,
        generate_summary: bool = True,
        **encode_kwargs,
    ):
        """
        Initialize the JMTEB v2.0 evaluator.

        Args:
            model: JMTEBModel instance to evaluate
            tasks: Single task or iterable of tasks to evaluate (e.g. the result of get_jmteb_tasks())
            save_path: Path to save summary.json (MTEB handles result caching)
            batch_size: Default batch size for encoding
            task_batch_sizes: Per-task batch size overrides
            cache_path: Path for MTEB's result cache
            overwrite_cache: Re-evaluate tasks even if results exist in the cache
            generate_summary: Write summary.json to save_path
            **encode_kwargs: Additional encoding keyword arguments
        """
        self.model = model
        self.tasks = [tasks] if isinstance(tasks, AbsTask) else list(tasks)
        self.save_path = Path(save_path) if save_path else None
        self.batch_size = batch_size
        self.task_batch_sizes = task_batch_sizes or {}
        self.cache_path = cache_path or "./cached_results"
        self.overwrite_cache = overwrite_cache
        self.generate_summary = generate_summary
        self.encode_kwargs = encode_kwargs

        # Create save directory if needed
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)

    @property
    def _write_summary(self) -> bool:
        return self.save_path is not None and self.generate_summary

    def _get_batch_size(self, task_name: str) -> int:
        """Get batch size for a specific task."""
        return self.task_batch_sizes.get(task_name, self.batch_size)

    def _extract_main_score(self, task_result: TaskResult, task_name: str) -> float | None:
        """Extract main score from MTEB task result."""
        split = SUMMARY_SPLITS.get(task_name, "test")
        if split not in task_result.scores:
            # e.g. MIRACL tasks are only evaluated on the dev split
            split = next(iter(task_result.scores), None)

        if split is not None and len(task_result.scores[split]) > 0:
            return task_result.scores[split][0].get("main_score")
        return None

    def _update_summary(self, task: AbsTask, task_result: TaskResult, eval_time: float, summary: dict):
        """Update summary with task result."""
        task_name = task.metadata.name
        main_score = self._extract_main_score(task_result, task_name)
        if main_score is None:
            logger.warning(f"No main score found for {task_name}; skipped in summary")
            return

        task_category = task.metadata.type
        summary.setdefault(task_category, {})[_get_task_key(task_name)] = {
            "main_metric": task.metadata.main_score,
            "main_score": main_score * 100,  # Convert to percentage
            "eval_time (s)": "%.2f" % eval_time,
        }

    def run(self) -> list[ModelResult] | None:
        """
        Run evaluation on all tasks.

        Returns:
            List of MTEB results objects (one per task), or None if no results
        """
        logger.info(f"Starting JMTEB v2.0 evaluation on {len(self.tasks)} tasks")
        logger.info(f"Tasks: {[task.metadata.name for task in self.tasks]}")

        if self._write_summary:
            logger.info(f"Summary will be saved to: {self.save_path}/summary.json")

        # Load existing summary
        summary = {}
        if self._write_summary:
            summary = load_summary(str(self.save_path))
            if summary:
                logger.info(f"Loaded existing summary from {self.save_path}/summary.json")

        all_results = []
        results_summary = []
        overwrite_strategy = "always" if self.overwrite_cache else "only-missing"

        # Evaluate each task
        for idx, task in enumerate(self.tasks, 1):
            task_name = task.metadata.name
            batch_size = self._get_batch_size(task_name)

            logger.info(f"\n[{idx}/{len(self.tasks)}] Task: {task_name} (batch_size={batch_size})")
            logger.info("-" * 80)

            start_time = time.time()

            encode_kwargs = {
                "batch_size": batch_size,
                **self.encode_kwargs,
            }

            # MTEB handles all caching automatically
            results = mteb.evaluate(
                model=self.model,
                tasks=task,
                encode_kwargs=encode_kwargs,
                cache=ResultCache(cache_path=self.cache_path),
                overwrite_strategy=overwrite_strategy,
            )

            elapsed_time = time.time() - start_time
            all_results.append(results)

            logger.info(f"✓ Completed: {task_name} (time: {elapsed_time:.2f}s)")
            results_summary.append((task_name, "✓ Success"))

            # Update summary
            if self._write_summary:
                self._update_summary(task, results.task_results[0], elapsed_time, summary)
                logger.info(f"Summary updated for {task_name}")
                # Save after each task
                save_summary(summary, str(self.save_path))
                logger.info(f"Summary saved to: {self.save_path}/summary.json")

        # Save final summary
        if self._write_summary:
            if summary:
                save_summary(summary, str(self.save_path))
                logger.info(f"Final summary saved to: {self.save_path}/summary.json")
            else:
                logger.warning("No summary data to save (summary dict is empty)")

        # Print final summary
        self._print_summary(results_summary)

        # Return list of all results or None if empty
        return all_results if all_results else None

    def _print_summary(self, results_summary: list[tuple[str, str]]):
        """Print evaluation summary."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total tasks: {len(self.tasks)}")

        successful = sum(1 for _, status in results_summary if "✓" in status)
        failed = sum(1 for _, status in results_summary if "✗" in status)

        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")

        if self._write_summary:
            logger.info(f"\nSummary saved to: {self.save_path}/summary.json")
        logger.info(f"MTEB cache: {self.cache_path}")

        logger.info("=" * 80)

        # Print detailed results
        if failed > 0:
            logger.info("\nDetailed Results:")
            for task_name, status in results_summary:
                if "✗" in status:
                    logger.info(f"  {status}")
            logger.info("=" * 80)
