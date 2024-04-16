# JMTEB: Japanese Massive Text Embedding Benchmark

[JMTEB](https://huggingface.co/datasets/sbintuitions/JMTEB) is a benchmark for evaluating Japanese text embedding models. It consists of 5 tasks.

This is an easy-to-use evaluation script designed for JMTEB evaluation.

## Quick start

```bash
git clone git@github.com:sbintuitions/JMTEB
cd JMTEB
poetry install
poetry run pytest tests
```

The following command evaluate the specified model on the all the tasks in JMTEB.

```bash
poetry run python -m jmteb \
  --embedder SentenceBertEmbedder \
  --embedder.model_name_or_path "<model_name_or_path>" \
  --save_dir "output/<model_name_or_path>"
```

> [!NOTE]
> In order to gurantee the robustness of evaluation, a validation dataset is mandatorily required for hyperparameter tuning.
> For a dataset that doesn't have a validation set, we set the validation set the same as the test set.

By default, the evaluation tasks are read from `src/jmteb/configs/jmteb.jsonnet`.
If you want to evaluate the model on a specific task, you can specify the task via `--evaluators` option with the task config.

```bash
poetry run python -m jmteb \
  --evaluators "src/configs/tasks/jsts.jsonnet" \
  --embedder SentenceBertEmbedder \
  --embedder.model_name_or_path "<model_name_or_path>" \
  --save_dir "output/<model_name_or_path>"
```

> [!NOTE]
> Some tasks (e.g., AmazonReviewClassification in classification, JAQKET and Mr.TyDi-ja in retrieval, esci in reranking) are time-consuming and memory-consuming. Heavy retrieval tasks take hours to encode the large corpus, and use much memory for the storage of such vectors. If you want to exclude them, add `--eval_exclude "['amazon_review_classification', 'mrtydi', 'jaqket', 'esci']"`.
