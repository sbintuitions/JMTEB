model=$1

echo "Running model: $model"

echo "start"
date "+%Y-%m-%d %H:%M:%S"
echo ""

# Data parallel
poetry run python -m jmteb \
  --embedder DataParallelSentenceBertEmbedder \
  --embedder.model_name_or_path "$model" \
  --embedder.model_kwargs '{"torch_dtype": "torch.float16"}' \
  --embedder.device cuda \
  --save_dir "results/${model//\//_}" \
  --overwrite_cache false \
  --evaluators src/jmteb/configs/jmteb.jsonnet

echo ""
date "+%Y-%m-%d %H:%M:%S"
echo "end"