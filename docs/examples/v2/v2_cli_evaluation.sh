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
  --prompt_profile src/jmteb/configs/prompts/e5.yaml \
  --save_path results_v2 \
  --batch_size 64

# Evaluate with per-task batch sizes
# batch_sizes.yaml is not included in the repository; create it beforehand.
# It maps task names to batch sizes (see "Batch Size Configuration" in README.md), e.g.
#   JSTS: 128
#   MIRACLRetrieval: 16
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

# Evaluate with BF16 precision
# Use bf16 rather than fp16 for models trained in bf16 such as ruri-v3: fp16 can overflow on long texts.
# Results are cached per model name regardless of precision, so use separate save/cache paths
# to avoid reusing the fp32 results of the commands above.
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --bf16 true \
  --save_path results_v2_bf16 \
  --cache_path cached_results_bf16 \
  --batch_size 64

# Overwrite existing cached results
python -m jmteb.v2 \
  --model_name cl-nagoya/ruri-v3-30m \
  --overwrite_cache true \
  --save_path results_v2 \
  --batch_size 32
