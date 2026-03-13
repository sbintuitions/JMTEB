"""
Basic JMTEB v2.0 evaluation example.

This script demonstrates how to evaluate a model on JMTEB v2.0 tasks using the MTEB framework.
"""

from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks

# Create model from HuggingFace (using small model for faster testing)
model = JMTEBModel.from_sentence_transformer(
    model_name_or_path="cl-nagoya/ruri-v3-30m",
)

# Get all JMTEB tasks (or specify task_names for specific tasks)
tasks = get_jmteb_tasks()

# Create evaluator
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    save_path="results_v2/ruri-v3-30m",
    batch_size=32,
)

# Run evaluation
results = evaluator.run()

print("Evaluation complete! Check results_v2/ruri-v3-30m/ for detailed results and summary.json")
