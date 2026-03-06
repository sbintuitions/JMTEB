# Using Custom Datasets with JMTEB

This guide explains how to evaluate embedding models on your own datasets using the JMTEB framework.

## Overview

JMTEB v2 is built on top of MTEB (Massive Text Embedding Benchmark), which provides a flexible framework for creating custom evaluation tasks. You can create custom tasks for your own datasets and evaluate them using JMTEB's evaluation infrastructure.

## Quick Start

Here's a simple example of evaluating a custom retrieval dataset:

```python
from mteb import AbsTaskRetrieval
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator

# 1. Define your custom task
class MyCustomRetrieval(AbsTaskRetrieval):
    metadata = {
        "name": "MyCustomRetrieval",
        "type": "Retrieval",
        "description": "My custom retrieval dataset",
        "main_score": "ndcg_at_10",
    }
    
    def load_data(self, **kwargs):
        # Load your dataset here
        # Return format: dict with splits (train/dev/test) containing:
        # - queries: dict[str, str]  # query_id -> query_text
        # - corpus: dict[str, dict]  # doc_id -> {"text": doc_text}
        # - relevant_docs: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance_score}
        
        queries = {
            "q1": "What is machine learning?",
            "q2": "How does deep learning work?",
        }
        
        corpus = {
            "d1": {"text": "Machine learning is a subset of AI..."},
            "d2": {"text": "Deep learning uses neural networks..."},
            "d3": {"text": "Artificial intelligence encompasses..."},
        }
        
        relevant_docs = {
            "q1": {"d1": 1, "d3": 1},
            "q2": {"d2": 1},
        }
        
        return {
            "test": {
                "queries": queries,
                "corpus": corpus,
                "relevant_docs": relevant_docs,
            }
        }

# 2. Create your model
model = JMTEBModel.from_sentence_transformer("your-model-name")

# 3. Instantiate the custom task
custom_task = MyCustomRetrieval()

# 4. Evaluate
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=[custom_task],
    save_path="results/custom_evaluation",
    batch_size=32,
)

results = evaluator.run()
```

## Task Types

MTEB supports several task types. Here's how to create each:

### 1. Retrieval Task

```python
from mteb import AbsTaskRetrieval

class MyRetrieval(AbsTaskRetrieval):
    metadata = {
        "name": "MyRetrieval",
        "type": "Retrieval",
        "description": "Description of your retrieval task",
        "main_score": "ndcg_at_10",  # or "map", "recall_at_k", etc.
    }
    
    def load_data(self, **kwargs):
        return {
            "test": {
                "queries": dict[str, str],
                "corpus": dict[str, dict[str, str]],
                "relevant_docs": dict[str, dict[str, int]],
            }
        }
```

### 2. Classification Task

```python
from mteb import AbsTaskClassification

class MyClassification(AbsTaskClassification):
    metadata = {
        "name": "MyClassification",
        "type": "Classification",
        "description": "Description of your classification task",
        "main_score": "accuracy",
    }
    
    def load_data(self, **kwargs):
        return {
            "test": [
                {"text": "Sample text 1", "label": 0},
                {"text": "Sample text 2", "label": 1},
                # ...
            ]
        }
```

### 3. Clustering Task

```python
from mteb import AbsTaskClustering

class MyClustering(AbsTaskClustering):
    metadata = {
        "name": "MyClustering",
        "type": "Clustering",
        "description": "Description of your clustering task",
        "main_score": "v_measure",
    }
    
    def load_data(self, **kwargs):
        return {
            "test": [
                {"sentence": "Text 1", "label": 0},
                {"sentence": "Text 2", "label": 1},
                {"sentence": "Text 3", "label": 0},
                # ...
            ]
        }
```

### 4. STS (Semantic Textual Similarity) Task

```python
from mteb import AbsTaskSTS

class MySTS(AbsTaskSTS):
    metadata = {
        "name": "MySTS",
        "type": "STS",
        "description": "Description of your STS task",
        "main_score": "cosine_spearman",
    }
    
    def load_data(self, **kwargs):
        return {
            "test": [
                {"sentence1": "Text A", "sentence2": "Text B", "score": 0.8},
                {"sentence1": "Text C", "sentence2": "Text D", "score": 0.3},
                # ...
            ]
        }
```

### 5. Reranking Task

```python
from mteb import AbsTaskReranking

class MyReranking(AbsTaskReranking):
    metadata = {
        "name": "MyReranking",
        "type": "Reranking",
        "description": "Description of your reranking task",
        "main_score": "map",
    }
    
    def load_data(self, **kwargs):
        return {
            "test": {
                "queries": dict[str, str],
                "corpus": dict[str, dict[str, str]],
                "relevant_docs": dict[str, dict[str, int]],
            }
        }
```

### 6. Pair Classification Task

```python
from mteb import AbsTaskPairClassification

class MyPairClassification(AbsTaskPairClassification):
    metadata = {
        "name": "MyPairClassification",
        "type": "PairClassification",
        "description": "Description of your pair classification task",
        "main_score": "cosine_ap",
    }
    
    def load_data(self, **kwargs):
        return {
            "test": [
                {"sent1": "Text 1", "sent2": "Text 2", "labels": 1},
                {"sent1": "Text 3", "sent2": "Text 4", "labels": 0},
                # ...
            ]
        }
```

## Loading Data from Files

You can load your dataset from various formats:

### From JSONL Files

```python
import json
from mteb import AbsTaskRetrieval

class MyRetrieval(AbsTaskRetrieval):
    metadata = {
        "name": "MyRetrieval",
        "type": "Retrieval",
        "description": "My custom retrieval task",
        "main_score": "ndcg_at_10",
    }
    
    def load_data(self, **kwargs):
        # Load queries
        queries = {}
        with open("data/queries.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                queries[item["id"]] = item["text"]
        
        # Load corpus
        corpus = {}
        with open("data/corpus.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                corpus[item["id"]] = {"text": item["text"]}
        
        # Load relevance judgments
        relevant_docs = {}
        with open("data/qrels.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                query_id = item["query_id"]
                if query_id not in relevant_docs:
                    relevant_docs[query_id] = {}
                relevant_docs[query_id][item["doc_id"]] = item["score"]
        
        return {
            "test": {
                "queries": queries,
                "corpus": corpus,
                "relevant_docs": relevant_docs,
            }
        }
```

### From Hugging Face Datasets

```python
from datasets import load_dataset
from mteb import AbsTaskClassification

class MyHFClassification(AbsTaskClassification):
    metadata = {
        "name": "MyHFClassification",
        "type": "Classification",
        "description": "Classification task from HF dataset",
        "main_score": "accuracy",
    }
    
    def load_data(self, **kwargs):
        # Load from Hugging Face
        dataset = load_dataset("your-username/your-dataset")
        
        # Convert to MTEB format
        test_data = [
            {"text": item["text"], "label": item["label"]}
            for item in dataset["test"]
        ]
        
        return {"test": test_data}
```

## Advanced: Multiple Splits

You can provide train/validation/test splits:

```python
def load_data(self, **kwargs):
    return {
        "train": [...],
        "validation": [...],
        "test": [...],
    }
```

## Evaluating Multiple Custom Tasks

```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator

# Create model
model = JMTEBModel.from_sentence_transformer("your-model")

# Create multiple custom tasks
task1 = MyRetrieval()
task2 = MyClassification()
task3 = MySTS()

# Evaluate all tasks
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=[task1, task2, task3],
    save_path="results/custom_evaluation",
    batch_size=32,
)

results = evaluator.run()
```

## Combining Custom Tasks with JMTEB Tasks

You can evaluate custom tasks alongside standard JMTEB tasks:

```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator, get_jmteb_tasks

# Create model
model = JMTEBModel.from_sentence_transformer("your-model")

# Get some JMTEB tasks
jmteb_tasks = get_jmteb_tasks(task_names=["JSTS", "JSICK"])

# Add your custom task
custom_task = MyCustomRetrieval()

# Evaluate both
evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=jmteb_tasks + [custom_task],
    save_path="results/combined_evaluation",
    batch_size=32,
)

results = evaluator.run()
```

## Complete Example: Custom Japanese Retrieval Task

```python
import json
from mteb import AbsTaskRetrieval
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator

class JapaneseQARetrieval(AbsTaskRetrieval):
    """Custom Japanese QA retrieval task."""
    
    metadata = {
        "name": "JapaneseQARetrieval",
        "type": "Retrieval",
        "description": "Japanese question answering retrieval",
        "main_score": "ndcg_at_10",
        "languages": ["jpn"],
    }
    
    def load_data(self, **kwargs):
        # Example: Load your Japanese QA data
        queries = {
            "q1": "日本の首都はどこですか？",
            "q2": "富士山の高さは？",
        }
        
        corpus = {
            "d1": {"text": "東京は日本の首都です。"},
            "d2": {"text": "富士山は標高3,776メートルの日本最高峰の山です。"},
            "d3": {"text": "大阪は日本の主要都市の一つです。"},
        }
        
        relevant_docs = {
            "q1": {"d1": 1},
            "q2": {"d2": 1},
        }
        
        return {
            "test": {
                "queries": queries,
                "corpus": corpus,
                "relevant_docs": relevant_docs,
            }
        }

# Evaluate
model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-base")
task = JapaneseQARetrieval()

evaluator = JMTEBV2Evaluator(
    model=model,
    tasks=[task],
    save_path="results/japanese_qa",
    batch_size=32,
)

results = evaluator.run()
print(f"NDCG@10: {results['JapaneseQARetrieval']['test']['ndcg_at_10']}")
```

## Tips and Best Practices

1. **Use descriptive task names**: Make sure your task name clearly describes what it evaluates

2. **Provide metadata**: Include a clear description and specify the main evaluation metric

3. **Handle data loading errors**: Add error handling in your `load_data()` method

4. **Test with small data first**: Start with a small subset to ensure your task works correctly

5. **Follow MTEB conventions**: Use the same data format as standard MTEB tasks for consistency

6. **Document your task**: Add comments explaining the dataset and evaluation setup

## Further Resources

- [MTEB Documentation](https://github.com/embeddings-benchmark/mteb)
- [MTEB Task Examples](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/tasks)
- [JMTEB Example Scripts](../examples/)

## Need Help?

If you encounter issues:

1. Check that your data format matches MTEB requirements
2. Look at existing JMTEB task implementations for reference
3. Ensure your task metadata is complete and correct
4. Test with a small dataset first before scaling up
