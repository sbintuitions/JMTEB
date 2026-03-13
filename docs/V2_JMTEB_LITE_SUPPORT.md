# JMTEB-lite Support

JMTEB-lite now has the same level of support as JMTEB in the codebase.

## What is JMTEB-lite?

JMTEB-lite is a lightweight version of JMTEB that provides:
- **~5x faster evaluation** compared to full JMTEB
- **0.97 Spearman correlation** with full JMTEB results
- **Same 28 tasks** but with reduced corpus sizes for retrieval and reranking tasks
- Ideal for agile development and quick evaluation iterations

The lightweight tasks (with reduced corpus sizes):
- Retrieval: JaqketRetrievalLite, MrTyDiJaRetrievalLite, JaCWIRRetrievalLite, MIRACLJaRetrievalLite
- Reranking: JQaRARerankingLite, JaCWIRRerankingLite

## Features Added

### 1. Task List Constant
```python
from jmteb.v2 import JMTEB_LITE_TASKS

# List of all 28 JMTEB-lite task names
print(f"JMTEB-lite has {len(JMTEB_LITE_TASKS)} tasks")
```

### 2. Task Retrieval Function
```python
from jmteb.v2 import get_jmteb_lite_tasks

# Get all JMTEB-lite tasks
tasks = get_jmteb_lite_tasks()

# Get specific tasks
tasks = get_jmteb_lite_tasks(task_names=["JSTS", "JSICK"])

# Get tasks by type
tasks = get_jmteb_lite_tasks(task_types=["Retrieval", "Classification"])
```

### 3. Benchmark Function (Already Existed)
```python
from jmteb.v2 import get_jmteb_lite_benchmark

benchmark = get_jmteb_lite_benchmark()
print(f"JMTEB-lite benchmark: {len(benchmark.tasks)} tasks")
```

## Usage Examples

### Example 1: Quick Evaluation with JMTEB-lite
```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator, get_jmteb_lite_tasks

# Create model
model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-base")

# Get JMTEB-lite tasks (for faster evaluation)
tasks = get_jmteb_lite_tasks()

# Create evaluator
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    save_path="results_lite/ruri-base",
    batch_size=32,
)

# Run evaluation (much faster!)
results = evaluator.run()
```

### Example 2: Filter Specific Task Types
```python
from jmteb.v2 import get_jmteb_lite_tasks

# Get only retrieval tasks for quick testing
retrieval_tasks = get_jmteb_lite_tasks(task_types=["Retrieval"])

# Get only STS tasks
sts_tasks = get_jmteb_lite_tasks(task_types=["STS"])
```

### Example 3: Using the Benchmark Directly
```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator, get_jmteb_lite_benchmark

model = JMTEBModel.from_sentence_transformer("model-name")

# Get all JMTEB-lite tasks via benchmark
benchmark = get_jmteb_lite_benchmark()

evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=benchmark.tasks,
    save_path="results_lite/model-name",
)

results = evaluator.run()
```

## Comparison with JMTEB

| Feature | JMTEB | JMTEB-lite |
|---------|-------|------------|
| Number of Tasks | 28 | 28 (same tasks) |
| Task Names | `JMTEB_TASKS` | `JMTEB_LITE_TASKS` |
| Get Tasks Function | `get_jmteb_tasks()` | `get_jmteb_lite_tasks()` |
| Get Benchmark Function | `get_jmteb_benchmark()` | `get_jmteb_lite_benchmark()` |
| Evaluation Speed | Baseline | ~5x faster |
| Corpus Size | Full | Reduced (retrieval/reranking) |
| Correlation | 1.0 (baseline) | 0.97 Spearman |
| Use Case | Final evaluation | Agile development, quick testing |

## Files Modified

1. **src/jmteb/v2/tasks.py**
   - Added `JMTEB_LITE_TASKS` constant
   - Added `get_jmteb_lite_tasks()` function with filtering support

2. **src/jmteb/v2/__init__.py**
   - Exported `JMTEB_LITE_TASKS`
   - Exported `get_jmteb_lite_tasks`

## Testing

All features have been tested and verified:
- ✓ Task list constants work
- ✓ Task retrieval functions work
- ✓ Task filtering by names works
- ✓ Task filtering by types works
- ✓ Benchmark retrieval works
- ✓ Integration with existing evaluator works

## Reference

- [JMTEB-lite on Hugging Face](https://huggingface.co/datasets/sbintuitions/JMTEB-lite)
- Example script: `docs/examples/v2_jmteb_lite.py`
