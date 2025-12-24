#!/bin/bash
# JMTEB v2.0 CLI evaluation examples

# Basic evaluation - all tasks (using small model for faster testing)
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --save_path results_v2 \
  --batch_size 32

# Evaluate specific tasks only
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --include JSTS JSICK JaqketRetrieval \
  --save_path results_v2 \
  --batch_size 32

# Evaluate with prompts (e.g., for E5 models)
python -m jmteb.v2 \
  --model_name intfloat/multilingual-e5-base \
  --prompt_profile prompts/e5.yaml \
  --save_path results_v2 \
  --batch_size 64

# Evaluate with per-task batch sizes
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-large \
  --task_batch_sizes batch_sizes.yaml \
  --save_path results_v2

# Evaluate only retrieval tasks
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --task_types Retrieval \
  --save_path results_v2 \
  --batch_size 32

# Evaluate with FP16 precision
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --fp16 true \
  --save_path results_v2 \
  --batch_size 64

# Overwrite existing cached results
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --overwrite_cache true \
  --save_path results_v2 \
  --batch_size 32
