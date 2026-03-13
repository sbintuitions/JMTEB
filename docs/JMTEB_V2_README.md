# JMTEB v2.0 Documentation

JMTEB v2.0 is a modernized implementation of the Japanese Massive Text Embedding Benchmark (JMTEB) built on top of the [MTEB](https://github.com/embeddings-benchmark/mteb) framework. It provides a cleaner API, better performance, and seamless integration with the broader MTEB ecosystem.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Python API](#python-api)
  - [Command-Line Interface](#command-line-interface)
- [Core Components](#core-components)
  - [JMTEBModel](#jmtebmodel)
  - [JMTEBV2Evaluator](#jmtebv2evaluator)
  - [Tasks](#tasks)
- [Task Coverage](#task-coverage)
- [Configuration](#configuration)
  - [Prompts](#prompts)
  - [Batch Sizes](#batch-sizes)
- [Advanced Usage](#advanced-usage)
  - [Task Filtering](#task-filtering)
  - [Using JMTEB-lite](#using-jmteb-lite)
  - [Wrapping v1 Embedders](#wrapping-v1-embedders)
  - [Using MTEB's Model Loader](#using-mtebs-model-loader)
- [Results Format](#results-format)
- [Architecture](#architecture)
- [Known Considerations](#known-considerations)
- [Migration from v1](#migration-from-v1)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Features

- **MTEB Integration**: Leverages MTEB's robust evaluation framework for standardized benchmarking
- **Simplified API**: Clean, Pythonic interface for running evaluations
- **28 Japanese Tasks**: Complete coverage of all JMTEB benchmark tasks
- **JMTEB-lite Support**: ~5x faster evaluation with lightweight task variants
- **Automatic Caching**: Smart result caching via MTEB to avoid redundant computation
- **Per-task Batch Sizes**: Optimize memory usage for different tasks
- **Prompt Configuration**: YAML-based prompt templates for instruction-tuned models
- **Summary Generation**: Automatic generation of `summary.json` with organized results
- **Backward Compatibility**: Supports wrapping v1.x embedders for seamless migration

---

## Installation

JMTEB v2.0 is now the default. Install with:

```bash
# Using poetry (recommended)
poetry install

# Using pip
pip install jmteb
```

For v1.x backward compatibility (includes legacy dependencies):

```bash
# Poetry
poetry install --extras v1

# Pip
pip install jmteb[v1]
```

---

## Quick Start

### Python API

```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks

# Create model
model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-base")

# Get tasks (all JMTEB tasks)
tasks = get_jmteb_tasks()

# Create evaluator and run
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    save_path="results_v2/ruri-base",
    batch_size=32,
)
results = evaluator.run()

# Results and summary.json are saved to results_v2/ruri-base/
```

### Command-Line Interface

```bash
# Basic evaluation (all tasks)
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-base \
  --save_path results_v2

# Evaluate specific tasks
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-base \
  --include JSTS JSICK JaqketRetrieval \
  --save_path results_v2

# With prompt configuration (for models like E5)
python -m jmteb.v2 \
  --model_name intfloat/multilingual-e5-base \
  --prompt_profile src/jmteb/configs/prompts/e5.yaml \
  --save_path results_v2

# With FP16 precision
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-base \
  --fp16 true \
  --save_path results_v2
```

---

## Core Components

### JMTEBModel

The `JMTEBModel` class is an adapter that bridges various model types with MTEB's evaluation system.

#### Creating from SentenceTransformer

```python
from jmteb.v2 import JMTEBModel

# Basic usage
model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-base")

# With device specification
model = JMTEBModel.from_sentence_transformer(
    "cl-nagoya/ruri-base",
    device="cuda:0"
)

# With model kwargs (e.g., precision)
import torch
model = JMTEBModel.from_sentence_transformer(
    "cl-nagoya/ruri-base",
    model_kwargs={"torch_dtype": torch.float16}
)

# With prompts
model = JMTEBModel.from_sentence_transformer(
    "intfloat/multilingual-e5-base",
    prompts={"query": "query: ", "passage": "passage: "}
)
```

#### Creating from JMTEB v1 Embedder

```python
from jmteb.embedders import SentenceBertEmbedder
from jmteb.v2 import JMTEBModel

# Create v1 embedder
v1_embedder = SentenceBertEmbedder(model_name_or_path="cl-nagoya/ruri-base")

# Wrap for v2
model = JMTEBModel.from_jmteb_embedder(v1_embedder)
```

#### Creating from MTEB

```python
from jmteb.v2 import JMTEBModel

# Use MTEB's model loading
model = JMTEBModel.from_mteb("sentence-transformers/all-MiniLM-L6-v2")
```

### JMTEBV2Evaluator

The `JMTEBV2Evaluator` class orchestrates the evaluation process.

```python
from jmteb.v2 import JMTEBV2Evaluator

evaluator = JMTEBV2Evaluator(
    model=model,                              # JMTEBModel instance
    tasks=tasks,                              # List of tasks or single task
    save_path="results_v2/model_name",        # Output directory
    batch_size=32,                            # Default batch size
    task_batch_sizes={"JSTS": 64},            # Per-task overrides
    cache_path="./cached_results",            # MTEB cache location
)

# Run evaluation
results = evaluator.run()
```

### Tasks

Task utilities for getting and filtering JMTEB tasks.

```python
from jmteb.v2.tasks import (
    get_jmteb_tasks,           # Get all JMTEB tasks
    get_jmteb_lite_tasks,      # Get JMTEB-lite tasks
    get_task_by_name,          # Get single task by name
    get_jmteb_benchmark,       # Get MTEB benchmark object
    get_jmteb_lite_benchmark,  # Get JMTEB-lite benchmark object
)

# Get all tasks
all_tasks = get_jmteb_tasks()

# Get specific tasks
sts_tasks = get_jmteb_tasks(task_names=["JSTS", "JSICK"])

# Get by task type
retrieval_tasks = get_jmteb_tasks(task_types=["Retrieval"])

# Get single task
jsts = get_task_by_name("JSTS")
```

---

## Task Coverage

JMTEB v2.0 includes 28 evaluation tasks across 5 categories:

### Classification (7 tasks)

| Task Name | Description |
|-----------|-------------|
| `AmazonReviewsClassification` | Amazon product review classification |
| `AmazonCounterfactualClassification` | Counterfactual classification |
| `MassiveIntentClassification` | Intent classification |
| `MassiveScenarioClassification` | Scenario classification |
| `JapaneseSentimentClassification` | Japanese sentiment analysis |
| `SIB200Classification` | Multi-topic classification |
| `WRIMEClassification` | Writer/reader emotion classification |

### Clustering (3 tasks)

| Task Name | Description |
|-----------|-------------|
| `LivedoorNewsClustering.v2` | News article clustering |
| `MewsC16JaClustering` | Multi-source news clustering |
| `SIB200ClusteringS2S` | Sentence-to-sentence clustering |

### Semantic Textual Similarity (2 tasks)

| Task Name | Description |
|-----------|-------------|
| `JSTS` | Japanese Semantic Textual Similarity |
| `JSICK` | Japanese SICK dataset |

### Retrieval (11 tasks)

| Task Name | Description |
|-----------|-------------|
| `JaqketRetrieval` | Quiz question retrieval |
| `MrTidyRetrieval` | Multi-domain retrieval |
| `JaGovFaqsRetrieval` | Government FAQ retrieval |
| `NLPJournalTitleAbsRetrieval.V2` | Paper title-abstract retrieval |
| `NLPJournalTitleIntroRetrieval.V2` | Paper title-intro retrieval |
| `NLPJournalAbsIntroRetrieval.V2` | Paper abstract-intro retrieval |
| `NLPJournalAbsArticleRetrieval.V2` | Paper abstract-article retrieval |
| `JaCWIRRetrieval` | Web information retrieval |
| `MIRACLRetrieval` | Multilingual retrieval |
| `MintakaRetrieval` | Knowledge base QA retrieval |
| `MultiLongDocRetrieval` | Long document retrieval |

### Reranking (5 tasks)

| Task Name | Description |
|-----------|-------------|
| `ESCIReranking` | E-commerce search reranking |
| `JQaRAReranking` | Question-answer reranking |
| `JaCWIRReranking` | Web information reranking |
| `MIRACLReranking` | Multilingual reranking |
| `MultiLongDocReranking` | Long document reranking |

---

## Configuration

### Prompts

Some models (e.g., E5, Ruri v3) require specific prompt templates. Use YAML configuration files:

```yaml
# src/jmteb/configs/prompts/e5.yaml
query: "query: "
passage: "passage: "
```

```yaml
# src/jmteb/configs/prompts/ruri-v3.yaml
query: "検索クエリ: "
passage: "検索文章: "
```

#### Using Prompts in Python

```python
from jmteb.v2 import JMTEBModel
from jmteb.v2.utils import load_prompts

prompts = load_prompts("src/jmteb/configs/prompts/e5.yaml")
model = JMTEBModel.from_sentence_transformer(
    "intfloat/multilingual-e5-base",
    prompts=prompts
)
```

#### Using Prompts via CLI

```bash
python -m jmteb.v2 \
  --model_name intfloat/multilingual-e5-base \
  --prompt_profile src/jmteb/configs/prompts/e5.yaml \
  --save_path results_v2
```

### Batch Sizes

Configure per-task batch sizes to optimize memory usage:

```yaml
# batch_sizes.yaml
JSTS: 128
JSICK: 128
JaqketRetrieval: 32
MIRACLRetrieval: 16
MultiLongDocRetrieval: 8
```

#### Using in Python

```python
from jmteb.v2 import JMTEBV2Evaluator
from jmteb.v2.utils import load_batch_sizes

batch_sizes = load_batch_sizes("batch_sizes.yaml")
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    batch_size=32,  # default
    task_batch_sizes=batch_sizes,  # per-task overrides
)
```

#### Using via CLI

```bash
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-base \
  --task_batch_sizes batch_sizes.yaml \
  --save_path results_v2
```

---

## Advanced Usage

### Task Filtering

```python
from jmteb.v2.tasks import get_jmteb_tasks

# By task name
tasks = get_jmteb_tasks(task_names=["JSTS", "JSICK", "JaqketRetrieval"])

# By task type
retrieval_tasks = get_jmteb_tasks(task_types=["Retrieval"])
classification_tasks = get_jmteb_tasks(task_types=["Classification"])

# Combine filters
sts_and_retrieval = get_jmteb_tasks(task_types=["STS", "Retrieval"])
```

### Using JMTEB-lite

JMTEB-lite provides ~5x faster evaluation with reduced corpus sizes while maintaining high correlation (0.97 Spearman) with full JMTEB results.

```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_lite_tasks

model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-base")

# Get JMTEB-lite tasks (reduced corpus sizes)
tasks = get_jmteb_lite_tasks()

evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    save_path="results_lite/ruri-base",
)
results = evaluator.run()
```

The following tasks have lightweight variants with reduced corpus sizes:
- `JaqketRetrievalLite`
- `MrTyDiJaRetrievalLite`
- `JaCWIRRetrievalLite`
- `MIRACLJaRetrievalLite`
- `JQaRARerankingLite`
- `JaCWIRRerankingLite`

### Wrapping v1 Embedders

Existing JMTEB v1 embedders can be wrapped for use with v2:

```python
from jmteb.embedders import SentenceBertEmbedder, OpenAIEmbedder
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator

# SentenceBERT embedder
v1_embedder = SentenceBertEmbedder(model_name_or_path="cl-nagoya/ruri-base")
model = JMTEBModel.from_jmteb_embedder(v1_embedder)

# OpenAI embedder
openai_embedder = OpenAIEmbedder(model="text-embedding-3-small")
model = JMTEBModel.from_jmteb_embedder(openai_embedder)

# Custom embedder (any class with encode method)
class CustomEmbedder:
    def encode(self, texts, batch_size=32):
        # Your logic here
        return embeddings

model = JMTEBModel.from_jmteb_embedder(CustomEmbedder())
```

### Using MTEB's Model Loader

```python
from jmteb.v2 import JMTEBModel

# Use MTEB's get_model for unified loading
model = JMTEBModel.from_mteb(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="main"
)
```

---

## Results Format

JMTEB v2.0 saves results in MTEB-compatible format.

### Per-task Result Files

Each task generates a JSON file (e.g., `JSTS.json`):

```json
{
  "validation": [
    {
      "main_score": 0.8234,
      "spearman": 0.8234,
      "pearson": 0.8156,
      "manhattan": 0.7923,
      "euclidean": 0.8012,
      "cosine": 0.8234
    }
  ]
}
```

### Summary File

A `summary.json` file is automatically generated:

```json
{
  "STS": {
    "jsts": {
      "main_metric": "cosine_spearman",
      "main_score": 82.34,
      "eval_time (s)": "12.34"
    },
    "jsick": {
      "main_metric": "cosine_spearman",
      "main_score": 78.56,
      "eval_time (s)": "8.21"
    }
  },
  "Retrieval": {
    "jaqket": {
      "main_metric": "ndcg_at_10",
      "main_score": 65.43,
      "eval_time (s)": "45.67"
    }
  }
}
```

---

## Architecture

```
JMTEB v2.0 Architecture
========================

┌─────────────────────────────────────────┐
│         User Interface                  │
│  (CLI: python -m jmteb.v2)              │
│  (Python API)                           │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│       JMTEBV2Evaluator                  │
│  - Task orchestration                   │
│  - Result caching (via MTEB)            │
│  - Summary generation                   │
│  - Progress tracking                    │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│         JMTEBModel                      │
│  (Adapter Layer)                        │
│  - Wraps v1 embedders                   │
│  - Wraps SentenceTransformer            │
│  - Provides encode() interface          │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│         MTEB Framework                  │
│  - Task execution                       │
│  - Metric computation                   │
│  - Result caching                       │
└─────────────────────────────────────────┘
```

### File Structure

```
src/jmteb/v2/
├── __init__.py      # Public API exports
├── __main__.py      # CLI entry point
├── adapters.py      # JMTEBModel adapter class
├── evaluator.py     # JMTEBV2Evaluator class
├── tasks.py         # Task definitions and utilities
└── utils.py         # Helper functions
```

---

## Known Considerations

### Corpus Size Differences

The following tasks use different corpus sizes in MTEB compared to JMTEB v1:

| Task | JMTEB v1 | MTEB |
|------|----------|------|
| `MultiLongDocRetrieval` | 10,000 docs | 30,000 docs |
| `MultiLongDocReranking` | 10,000 docs | 30,000 docs |

This may result in slight score differences between v1 and v2 for these tasks.

### Result Splits

Different tasks use different evaluation splits:

- **JSTS**: `validation` split
- **MultiLongDoc***: `dev` split
- **All others**: `test` split

---

## Migration from v1

See the [Migration Guide](MIGRATION_V2.md) for detailed instructions on migrating from JMTEB v1.x.

### Quick Comparison

| Feature | v1.x | v2.0 |
|---------|------|------|
| API Style | Multiple evaluator classes | Single evaluator |
| Dataset Loading | Manual | Automatic via MTEB |
| Caching | Basic | Robust MTEB caching |
| Summary | Manual | Automatic generation |
| Task Names | `jsts`, `jaqket` | `JSTS`, `JaqketRetrieval` |

### Task Name Mapping

```python
from jmteb.v2.tasks import convert_v1_task_name

convert_v1_task_name("jsts")    # Returns "JSTS"
convert_v1_task_name("jaqket")  # Returns "JaqketRetrieval"
```

---

## API Reference

### jmteb.v2

| Export | Type | Description |
|--------|------|-------------|
| `JMTEBModel` | Class | Model adapter for MTEB |
| `JMTEBV2Evaluator` | Class | Evaluation orchestrator |
| `JMTEB_TASKS` | List | All JMTEB task names |
| `JMTEB_LITE_TASKS` | List | All JMTEB-lite task names |
| `get_jmteb_tasks` | Function | Get JMTEB tasks with filtering |
| `get_jmteb_lite_tasks` | Function | Get JMTEB-lite tasks |
| `get_jmteb_benchmark` | Function | Get MTEB benchmark object |
| `get_jmteb_lite_benchmark` | Function | Get JMTEB-lite benchmark |
| `get_task_by_name` | Function | Get single task by name |
| `load_prompts` | Function | Load prompt YAML |
| `load_batch_sizes` | Function | Load batch size YAML |
| `save_results` | Function | Save results to JSON |
| `load_summary` | Function | Load summary.json |
| `save_summary` | Function | Save summary.json |

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | required | Model name or path |
| `--batch_size` | int | 32 | Default batch size |
| `--fp16` | bool | false | Use FP16 precision |
| `--bf16` | bool | false | Use BF16 precision |
| `--include` | list | None | Tasks to include |
| `--exclude` | list | None | Tasks to exclude |
| `--task_types` | list | None | Filter by task type |
| `--prompt_profile` | str | None | Path to prompt YAML |
| `--task_batch_sizes` | str | None | Path to batch size YAML |
| `--save_path` | str | `results_v2` | Output directory |
| `--overwrite_cache` | bool | false | Overwrite cached results |
| `--generate_summary` | bool | true | Generate summary.json |
| `--cache_path` | str | `./cached_results` | MTEB cache directory |

---

## Examples

See the [examples/v2](examples/v2/) directory for complete working examples:

- [`v2_basic_evaluation.py`](examples/v2/v2_basic_evaluation.py) - Basic Python API usage
- [`v2_cli_evaluation.sh`](examples/v2/v2_cli_evaluation.sh) - CLI usage patterns
- [`v2_jmteb_lite.py`](examples/v2/v2_jmteb_lite.py) - Using JMTEB-lite
- [`v2_mteb_model_loader.py`](examples/v2/v2_mteb_model_loader.py) - Using MTEB model loader

---

## Support

- **Migration Guide**: [docs/MIGRATION_V2.md](MIGRATION_V2.md)
- **Architecture Details**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **Custom Datasets**: [docs/V2_CUSTOM_DATASET_GUIDE.md](V2_CUSTOM_DATASET_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/sbintuitions/JMTEB/issues)
