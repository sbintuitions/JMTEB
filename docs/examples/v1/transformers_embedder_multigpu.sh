model=$1
# Number of GPUs in the node (default: 8)
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

echo "Running model: $model"

echo "start"
date "+%Y-%m-%d %H:%M:%S"
echo ""

MODEL_KWARGS='{"torch_dtype": "torch.bfloat16"}'

# embedder.batch_size is global batch size

poetry run torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=1 \
    src/jmteb/__main__.py --embedder TransformersEmbedder \
    --embedder.model_name_or_path "$model" \
    --embedder.pooling_mode cls \
    --embedder.batch_size 4096 \
    --embedder.model_kwargs "$MODEL_KWARGS" \
    --embedder.max_seq_length 512 \
    --save_dir "results/${model//\//_}" \
    --evaluators src/jmteb/configs/jmteb.jsonnet

echo ""
date "+%Y-%m-%d %H:%M:%S"
echo "end"
