# JMTEB v2.0 - MTEB-Powered Japanese Text Embedding Benchmark

JMTEB v2.0 is a major update to the Japanese Massive Text Embedding Benchmark that integrates with the [MTEB (Massive Text Embedding Benchmark)](https://github.com/embeddings-benchmark/mteb) framework.

## Overview

JMTEB v2.0 provides:

- 🌐 **MTEB Compatibility**: Integration with MTEB tools, leaderboards, and ecosystem
- 🚀 **MTEB as Evaluation Engine**: Leverage MTEB's robust framework
- 🎯 **Simpler API**: Cleaner, more intuitive interface
- 📊 **28 Japanese Datasets**: Comprehensive evaluation across 5 task types
- ⚡ **High Performance**: Efficient caching and batch processing

> [!IMPORTANT]
> The leaderboard is now hosted on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (General Purpose → Language-specific → Japanese). We no longer maintain a separate leaderboard in this repository.

## Quick Start

### Installation

**From source (Poetry):**

```bash
# Clone the repository
git clone https://github.com/sbintuitions/JMTEB.git
cd JMTEB

# Default (v2.0 with OpenAI support)
poetry install

# With v1.x support
poetry install --extras v1

# With everything
poetry install --all-extras
```

> [!NOTE]
> The package is not yet available on PyPI. Please install from source using the commands above.

### Basic Usage

```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks, get_jmteb_lite_benchmark

# Create model
model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-v3-30m")

# Get datasets - Full JMTEB
tasks = get_jmteb_tasks()  # All 28 JMTEB datasets

# Or use JMTEB-lite for faster evaluation
# lite_benchmark = get_jmteb_lite_benchmark()
# tasks = lite_benchmark.tasks

# Evaluate
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    save_path="results_v2"
)
results = evaluator.run()
```

### CLI Usage

```bash
# Evaluate all datasets
python -m jmteb.v2 --model_name cl-nagoya/ruri-v3-30m --save_path results_v2

# Evaluate specific datasets
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --include JSTS JSICK JaqketRetrieval \
  --save_path results_v2

# Use prompts (e.g., for Ruri-v3 models)
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --prompt_profile src/jmteb/configs/prompts/ruri-v3.yaml \
  --save_path results_v2
```

## Architecture

### Core Components

```
src/jmteb/v2/
├── __init__.py          # Main exports
├── __main__.py          # CLI entry point
├── adapters.py          # Model adapter (JMTEBModel)
├── evaluator.py         # Evaluation orchestrator
├── tasks.py             # Task definitions and utilities
└── utils.py             # Helper functions
```

### Key Classes

#### 1. JMTEBModel

Adapter that bridges models with MTEB's evaluation system.

```python
from jmteb.v2 import JMTEBModel

# From HuggingFace via SentenceTransformer
model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-v3-30m")

# From MTEB's unified model interface (recommended)
model = JMTEBModel.from_mteb("cl-nagoya/ruri-v3-30m")

```

#### 2. JMTEBV2Evaluator

Orchestrates evaluation across multiple tasks.

```python
from jmteb.v2 import JMTEBV2Evaluator

evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    save_path="results_v2",
    batch_size=32,
    task_batch_sizes={"JSTS": 128},  # Per-task overrides
    overwrite_cache=False,
    generate_summary=True,
)
results = evaluator.run()
```

#### 3. Task Utilities

Functions for working with JMTEB tasks.

```python
from jmteb.v2.tasks import (
    get_jmteb_benchmark,
    get_jmteb_tasks,
    get_task_by_name,
    get_task_category,
)

# Get all datasets
all_tasks = get_jmteb_tasks()

# Filter by task type
retrieval_tasks = get_jmteb_tasks(task_types=["Retrieval"])

# Get specific datasets
specific_tasks = get_jmteb_tasks(task_names=["JSTS", "JSICK"])

# Get full benchmark
benchmark = get_jmteb_benchmark()
```

## Task Coverage

JMTEB v2.0 includes 28 datasets across 5 task types.

### JMTEB vs JMTEB-lite

- **JMTEB (Full)**: Complete benchmark with full corpus sizes
- **JMTEB-lite**: Lightweight version with reduced corpus sizes for faster evaluation (~5x faster with high correlation to full JMTEB)

Both versions include the same 28 datasets:

### Classification (7 datasets)

- AmazonReviewsClassification
- AmazonCounterfactualClassification
- MassiveIntentClassification
- MassiveScenarioClassification
- JapaneseSentimentClassification
- SIB200Classification
- WRIMEClassification

### Clustering (3 datasets)

- LivedoorNewsClustering.v2
- MewsC16JaClustering
- SIB200ClusteringS2S

### STS (2 datasets)

- JSTS
- JSICK

### Retrieval (11 datasets)

- JaqketRetrieval (→ JaqketRetrievalLite in JMTEB-lite)
- MrTidyRetrieval (→ MrTyDiJaRetrievalLite in JMTEB-lite)
- JaGovFaqsRetrieval
- NLPJournalTitleAbsRetrieval.V2
- NLPJournalTitleIntroRetrieval.V2
- NLPJournalAbsIntroRetrieval.V2
- NLPJournalAbsArticleRetrieval.V2
- JaCWIRRetrieval (→ JaCWIRRetrievalLite in JMTEB-lite)
- MIRACLRetrieval (→ MIRACLJaRetrievalLite in JMTEB-lite)
- MintakaRetrieval
- MultiLongDocRetrieval

### Reranking (5 datasets)

- ESCIReranking
- JQaRAReranking (→ JQaRARerankingLite in JMTEB-lite)
- JaCWIRReranking (→ JaCWIRRerankingLite in JMTEB-lite)
- MIRACLReranking
- MultiLongDocReranking

## Features

### 1. Prompt Support

Configure prompts for models that require them (e.g., E5):

```yaml
# src/jmteb/configs/prompts/e5.yaml
query: "query: "
document: "passage: "
```

```python
from jmteb.v2.utils import load_prompts

prompts = load_prompts("src/jmteb/configs/prompts/e5.yaml")
model = JMTEBModel.from_sentence_transformer(
    "intfloat/multilingual-e5-base",
    prompts=prompts
)
```

### 2. Batch Size Configuration

Different tasks have varying memory requirements (e.g., retrieval tasks with long documents need smaller batches to avoid OOM), so per-task batch size configuration helps optimize performance.

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

### 3. Result Caching

Results are automatically cached to avoid re-evaluation:

```python
# First run: evaluates all tasks
evaluator.run()

# Second run: loads from cache
evaluator.run()  # Instant!

# Force re-evaluation
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=tasks,
    overwrite_cache=True
)
evaluator.run()
```

### 4. Summary Generation

Automatically generates `summary.json` with main scores:

```json
{
  "Classification": {
    "amazon_review_classification": {
      "main_metric": "accuracy",
      "main_score": 67.32,
      "eval_time (s)": "12.34"
    }
  },
  "STS": {
    "jsts": {
      "main_metric": "cosine_spearman",
      "main_score": 82.14,
      "eval_time (s)": "5.67"
    }
  }
}
```

### 5. Progress Tracking

Real-time progress updates during evaluation:

```
[1/28] Task: JSTS (batch_size=128)
--------------------------------------------------------------------------------
✓ Completed: JSTS (time: 5.67s)
  → Updated summary: STS/jsts = 82.14 (time: 5.67s)

[2/28] Task: JSICK (batch_size=128)
--------------------------------------------------------------------------------
✓ Loaded from cache: JSICK
  → Updated summary: STS/jsick = 76.89 (cached)
```

## Advanced Usage

### Custom Model Implementation

Implement your own model by following the encode interface:

```python
import numpy as np
from jmteb.v2 import JMTEBModel

class CustomModel:
    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        # Your encoding logic here
        embeddings = your_model.encode(sentences)
        return np.array(embeddings)

# Wrap for JMTEB v2
model = JMTEBModel(sentence_transformer=CustomModel())
```

### Dataset-Specific Evaluation

Evaluate subsets of datasets:

```python
from jmteb.v2.tasks import get_jmteb_tasks

# Only datasets from STS task
sts_tasks = get_jmteb_tasks(task_types=["STS"])

# Specific retrieval datasets
retrieval_tasks = get_jmteb_tasks(
    task_names=["JaqketRetrieval", "MIRACLRetrieval"]
)

# All classification datasets
classification_tasks = get_jmteb_tasks(task_types=["Classification"])
```

### Mixed Precision

Use FP16/BF16 for faster evaluation:

```python
import torch

model = JMTEBModel.from_sentence_transformer(
    "cl-nagoya/ruri-v3-30m",
    model_kwargs={"torch_dtype": torch.bfloat16}
)
```

Or via CLI:

```bash
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --bf16 true \
  --save_path results_v2
```

## Comparison with v1.x

| Feature | v1.x | v2.0 |
|---------|------|------|
| Evaluation Engine | Custom | MTEB |
| Dataset Count | 28 | 28 |
| Batch Configuration | Per-embedder | Global + Per-task |
| Prompt Support | Limited | Full |
| MTEB Compatibility | No | Yes |

## Output Structure

```
results_v2/
└── model_name/
    ├── JSTS.json                    # Individual dataset results
    ├── JSICK.json
    ├── JaqketRetrieval.json
    ├── ...
    └── summary.json                 # Aggregated summary
```

Each dataset result file contains:

```json
{
  "test": [
    {
      "main_score": 0.8214,
      "metric1": value1,
      "metric2": value2,
      ...
    }
  ]
}
```

## Migration from v1.x

See [MIGRATION_V2.md](./docs/MIGRATION_V2.md) for a comprehensive migration guide.

Quick comparison:

**v1.x:**

```python
from jmteb.embedders import SentenceBertEmbedder
embedder = SentenceBertEmbedder(model_name_or_path="cl-nagoya/ruri-base")
# ... manual evaluator setup
```

**v2.0:**

```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks

model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-v3-30m")
tasks = get_jmteb_tasks()
evaluator = JMTEBV2Evaluator(model=model, tasks=tasks)
evaluator.run()
```

## Performance Tips

1. **Use appropriate batch sizes**: Larger models need smaller batches
2. **Enable caching**: Don't overwrite unless necessary
3. **Use mixed precision**: BF16 can significantly speed up evaluation
4. **Filter datasets**: Evaluate only what you need for faster iteration
5. **Per-dataset batch sizes**: Optimize for memory requirements

## Examples

See the `docs/examples/v2/` directory for complete examples:

- `v2_basic_evaluation.py`: Basic usage
- `v2_cli_evaluation.sh`: CLI examples
- `v2_jmteb_lite.py`: JMTEB-lite usage
- `v2_mteb_model_loader.py`: Using MTEB's model loader

For v1.x examples, see `docs/examples/v1/`.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- MTEB >= 1.22.0
- sentence-transformers >= 5.0
- Other dependencies in `pyproject.toml`

## Citation

If you use JMTEB v2.0, please cite both JMTEB and MTEB:

```bibtex
@inproceedings{li2026jmteb,
    author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide and Kawahara, Daisuke},
    title = {{JMTEB and JMTEB-lite: Japanese Massive Text Embedding Benchmark and Its Lightweight Version}},
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = may,
    year = "2026",
    address = "Palma, Mallorca, Spain",
    publisher = "European Language Resources Association",
    note = "to appear",
}

@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022}
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}
```

## License

Same as JMTEB v1.x. See LICENSE file.

## Contributing

Contributions are welcome! Please:

1. Follow existing code style
2. Add tests for new features
3. Update documentation

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: See `docs/` directory
- **Migration Help**: See `docs/MIGRATION_V2.md`

## Acknowledgments

- MTEB team for the excellent evaluation framework
- All contributors to JMTEB v1.x
- Japanese NLP community for dataset contributions
