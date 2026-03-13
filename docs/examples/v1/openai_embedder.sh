model=$1

export OPENAI_API_KEY=<your_openai_api_key>

echo "Running OpenAI model: $model"

echo "start"
date "+%Y-%m-%d %H:%M:%S"
echo ""

poetry run python -m jmteb \
  --embedder OpenAIEmbedder \
  --embedder.model "$model" \
  --save_dir "results/${model//\//_}" \
  --overwrite_cache false \
  --evaluators src/jmteb/configs/jmteb.jsonnet

echo ""
date "+%Y-%m-%d %H:%M:%S"
echo "end"