"""
JMTEB v2.0 example using MTEB's get_model interface.

This example demonstrates using mteb.get_model() to load models,
which provides a unified interface for various model types.
"""

from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks

# Load model using MTEB's unified interface (recommended)
# This automatically handles model-specific configurations
model = JMTEBModel.from_mteb("sentence-transformers/all-MiniLM-L6-v2")

# You can also specify additional arguments
# model = JMTEBModel.from_mteb(
#     "intfloat/multilingual-e5-base",
#     revision="main",
#     device="cuda"
# )

# Get specific tasks for quick testing
tasks = get_jmteb_tasks(task_names=["JSTS", "JSICK"])

# Create evaluator
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    save_path="results_v2/all-MiniLM-L6-v2",
    batch_size=128,  # Can use larger batch for smaller models
)

# Run evaluation
print("Starting evaluation with MTEB-loaded model...")
results = evaluator.run()

print("\nEvaluation complete!")
print("Results saved to: results_v2/all-MiniLM-L6-v2/")
