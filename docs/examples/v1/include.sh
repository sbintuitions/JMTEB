model=$1

echo "Running model: $model"

echo "start"
date "+%Y-%m-%d %H:%M:%S"
echo ""

poetry run python -m jmteb \
  --embedder SentenceBertEmbedder \
  --embedder.model_name_or_path "$model" \
  --embedder.model_kwargs '{"torch_dtype": "torch.float16"}' \
  --embedder.device cuda \
  --save_dir "results/${model//\//_}" \
  --overwrite_cache false \
  --evaluators src/jmteb/configs/jmteb.jsonnet \
  --eval_include "['livedoor_news', 'esci']"

echo ""
date "+%Y-%m-%d %H:%M:%S"
echo "end"