# JMTEB v2.0 Migration Guide

This guide helps you migrate from JMTEB v1.x to v2.0.

> [!NOTE]
> For complete v2.0 documentation, see the main [README.md](../README.md). This guide focuses specifically on migration from v1.x.

## Quick Start

### Installation

JMTEB v2.0 is now the default. See [README.md](../README.md#installation) for installation instructions.

**TL;DR:**

```bash
# v2.0 only (default)
poetry install

# v2.0 + v1.x support
poetry install --extras v1
```

### Do I Need to Migrate?

**Yes, eventually.** While v1.x code will continue to work for now, the v1.x API is deprecated and will no longer be actively maintained. We recommend migrating to v2.0 for:

- Continued support and bug fixes
- New features and improvements
- Better performance and MTEB ecosystem integration

## Migration Examples

### Example 1: Basic STS Dataset Evaluation

<details>
<summary>v1.x Code</summary>

```python
from jmteb.embedders import SentenceBertEmbedder
from jmteb.evaluators import STSEvaluator
from datasets import load_dataset

# Create embedder
embedder = SentenceBertEmbedder(model_name_or_path="cl-nagoya/ruri-v3-30m")

# Load dataset and create evaluator
dataset = load_dataset("sbintuitions/JMTEB", name="jsts")
evaluator = STSEvaluator(val_dataset=dataset["validation"])

# Run evaluation
metrics = evaluator(embedder)
print(metrics)
```

</details>

<details open>
<summary>v2.0 Code</summary>

```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks

# Create model
model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-v3-30m")

# Get datasets
tasks = get_jmteb_tasks(task_names=["JSTS"])

# Create evaluator and run
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    save_path="results_v2/ruri-v3-30m"
)
results = evaluator.run()
```

</details>

**Key Differences:**

- No manual dataset loading required
- Single evaluator handles all tasks
- Automatic caching and summary generation

### Example 2: Multiple Datasets from Different Task Types

<details>
<summary>v1.x Code</summary>

```python
from jmteb.embedders import SentenceBertEmbedder
from jmteb.evaluators import STSEvaluator, RetrievalEvaluator, ClassificationEvaluator

embedder = SentenceBertEmbedder(model_name_or_path="cl-nagoya/ruri-v3-30m")

# Manual evaluation of each task
jsts_eval = STSEvaluator(...)
jsts_metrics = jsts_eval(embedder)

jaqket_eval = RetrievalEvaluator(...)
jaqket_metrics = jaqket_eval(embedder)

amazon_eval = ClassificationEvaluator(...)
amazon_metrics = amazon_eval(embedder)
```

</details>

<details open>
<summary>v2.0 Code</summary>

```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks

model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-v3-30m")

# Evaluate multiple datasets at once
tasks = get_jmteb_tasks(
    task_names=["JSTS", "JaqketRetrieval", "AmazonReviewsClassification"]
)

evaluator = JMTEBV2Evaluator(
    model=model, 
    tasks=tasks, 
    save_path="results_v2/ruri-v3-30m"
)
results = evaluator.run()
```

</details>

### Example 3: Wrapping v1 Embedders

You can wrap existing v1 embedders for use with v2:

```python
from jmteb.embedders import SentenceBertEmbedder
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks

# Create v1 embedder (your existing code)
v1_embedder = SentenceBertEmbedder(model_name_or_path="cl-nagoya/ruri-v3-30m")

# Wrap for v2
model = JMTEBModel.from_jmteb_embedder(v1_embedder)

# Use with v2 evaluator (all datasets)
tasks = get_jmteb_tasks()
evaluator = JMTEBV2Evaluator(model=model, tasks=tasks)
results = evaluator.run()
```

### Example 4: CLI Comparison

<table>
<tr>
<th>v1.x CLI</th>
<th>v2.0 CLI</th>
</tr>
<tr>
<td>

```bash
python -m jmteb \
  --embedder SentenceBertEmbedder \
  --embedder.model_name_or_path cl-nagoya/ruri-v3-30m \
  --save_dir results/ruri-v3-30m
```

</td>
<td>

```bash
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --save_path results_v2
```

</td>
</tr>
</table>

## Dataset Name Mapping

Dataset names have been updated to match MTEB conventions. Use the helper function for conversion:

```python
from jmteb.v2.tasks import convert_v1_task_name

v2_name = convert_v1_task_name("jsts")  # Returns "JSTS"
v2_name = convert_v1_task_name("jaqket")  # Returns "JaqketRetrieval"
```

<details>
<summary>Complete Dataset Name Mapping Table</summary>

| v1.x Name | v2.0 Name |
|-----------|-----------|
| `livedoor_news` | `LivedoorNewsClustering.v2` |
| `mewsc16` | `MewsC16JaClustering` |
| `jsts` | `JSTS` |
| `jsick` | `JSICK` |
| `jaqket` | `JaqketRetrieval` |
| `mrtydi` | `MrTidyRetrieval` |
| `jagovfaqs_22k` | `JaGovFaqsRetrieval` |
| `nlp_journal_title_abs` | `NLPJournalTitleAbsRetrieval.V2` |
| `nlp_journal_title_intro` | `NLPJournalTitleIntroRetrieval.V2` |
| `nlp_journal_abs_intro` | `NLPJournalAbsIntroRetrieval.V2` |
| `nlp_journal_abs_article` | `NLPJournalAbsArticleRetrieval.V2` |
| `jacwir_retrieval` | `JaCWIRRetrieval` |
| `miracl_retrieval` | `MIRACLRetrieval` |
| `esci` | `ESCIReranking` |
| `jqara` | `JQaRAReranking` |
| `jacwir_reranking` | `JaCWIRReranking` |
| `miracl_reranking` | `MIRACLReranking` |
| `amazon_review_classification` | `AmazonReviewsClassification` |
| `amazon_counterfactual_classification` | `AmazonCounterfactualClassification` |
| `massive_intent_classification` | `MassiveIntentClassification` |
| `massive_scenario_classification` | `MassiveScenarioClassification` |

See `src/jmteb/v2/tasks.py` for the complete mapping in code.

</details>

## Configuration Migration

### Prompts

**v2.0 uses YAML configuration files:**

```yaml
# src/jmteb/configs/prompts/e5.yaml
query: "query: "
passage: "passage: "
```

Load and use:

```python
from jmteb.v2.utils import load_prompts

prompts = load_prompts("src/jmteb/configs/prompts/e5.yaml")
model = JMTEBModel.from_sentence_transformer(
    "intfloat/multilingual-e5-base",
    prompts=prompts
)
```

Or via CLI:

```bash
python -m jmteb.v2 \
  --model_name intfloat/multilingual-e5-base \
  --prompt_profile src/jmteb/configs/prompts/e5.yaml \
  --save_path results_v2
```

### Batch Sizes

**v2.0 supports per-task batch size configuration:**

```yaml
# batch_sizes.yaml
JSTS: 128
JSICK: 128
JaqketRetrieval: 32
MIRACLRetrieval: 16
MultiLongDocRetrieval: 8
```

```python
from jmteb.v2.utils import load_batch_sizes

batch_sizes = load_batch_sizes("batch_sizes.yaml")
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    task_batch_sizes=batch_sizes
)
```

## Results Format Changes

### v1.x Results Structure

```json
{
  "spearman": 0.8234,
  "pearson": 0.8156
}
```

### v2.0 Results Structure

```json
{
  "validation": [
    {
      "main_score": 0.8234,
      "spearman": 0.8234,
      "pearson": 0.8156,
      ...
    }
  ]
}
```

**Key Changes:**

- Results organized by split (test/validation/dev)
- `main_score` field indicates the primary metric
- More detailed metrics included
- MTEB-compatible format

**Extracting Main Score:**

```python
# v2.0
with open("results_v2/model/JSTS.json") as f:
    results = json.load(f)
    main_score = results["validation"][0]["main_score"]
```

## Migration FAQ

### Do the results change between v1 and v2?

The evaluation logic is identical for most tasks, so results are directly comparable. Minor differences may occur for:

- `MultiLongDocRetrieval` and `MultiLongDocReranking` (corpus size differences)

### Can I use both v1 and v2 in the same project?

Yes! They are separate modules:

```python
# v1.x
from jmteb.embedders import SentenceBertEmbedder

# v2.0
from jmteb.v2 import JMTEBModel
```

### What happens to my v1 code when I install v2?

Nothing! v1 code continues to work as-is. Install v2 dependencies with:

```bash
poetry install --extras v1  # Includes both v1 and v2
```

### Can I convert v1 results to v2 format?

The underlying metrics are the same, so you can manually convert if needed. However, re-running with v2 is recommended for consistency.

### What about OpenAI/custom embedders?

They work in v2:

```python
# OpenAI
from jmteb.embedders import OpenAIEmbedder
from jmteb.v2 import JMTEBModel

embedder = OpenAIEmbedder(model="text-embedding-3-small")
model = JMTEBModel.from_jmteb_embedder(embedder)

# Custom
class CustomEmbedder:
    def encode(self, texts, batch_size=32):
        # Your logic
        return embeddings

model = JMTEBModel.from_jmteb_embedder(CustomEmbedder())
```

## Next Steps

1. **Read the main README**: See [README.md](../README.md) for complete v2.0 documentation
2. **Try the examples**: Check `examples/v2_*.py` for working code
3. **Start small**: Migrate one evaluation at a time
4. **Ask for help**: Open an issue if you encounter problems

## Benefits of Migrating

- 🚀 Simpler, cleaner API
- ⚡ Better performance and caching
- 🌐 Access to MTEB ecosystem
- 📊 Automatic summary generation
- 🔧 Per-task batch size configuration
- 🎯 Progress tracking and better logging

Happy migrating!
