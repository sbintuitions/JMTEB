"""
JMTEB-lite evaluation example.

This script demonstrates how to use JMTEB-lite for faster evaluation.
JMTEB-lite provides ~5x speedup with 0.97 Spearman correlation to full JMTEB.
"""

from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator, get_jmteb_lite_benchmark

# Create model from HuggingFace (using small model for faster testing)
model = JMTEBModel.from_sentence_transformer(
    model_name_or_path="cl-nagoya/ruri-v3-30m",
)

# Get JMTEB-lite benchmark (reduced corpus sizes)
lite_benchmark = get_jmteb_lite_benchmark()
print(f"JMTEB-lite contains {len(lite_benchmark.tasks)} tasks")

# Create evaluator
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=lite_benchmark.tasks,
    save_path="results_v2_lite/ruri-v3-30m",
    batch_size=32,
)

# Run evaluation (much faster than full JMTEB!)
results = evaluator.run()

print("\n" + "=" * 80)
print("JMTEB-lite evaluation complete!")
print("Results saved to: results_v2_lite/ruri-v3-30m/")
print("=" * 80)
print("\nNote: JMTEB-lite is ~5x faster than full JMTEB")
print("with 0.97 Spearman correlation to full JMTEB results.")
