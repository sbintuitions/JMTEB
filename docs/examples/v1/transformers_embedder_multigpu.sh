model=$1

echo "Running model: $model"

echo "start"
date "+%Y-%m-%d %H:%M:%S"
echo ""

MODEL_KWARGS="\{\'torch_dtype\':\'torch.bfloat16\'\}"

# embedder.batch_size is global batch size

torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=1 \
    src/jmteb/__main__.py --embedder TransformersEmbedder \
    --embedder.model_name_or_path ${MODEL_NAME} \
    --embedder.pooling_mode cls \
    --embedder.batch_size 4096 \
    --embedder.model_kwargs ${MODEL_KWARGS} \
    --embedder.max_seq_length 512 \
    --save_dir "results/${MODEL_NAME}" \
    --evaluators src/jmteb/configs/jmteb.jsonnet

echo ""
date "+%Y-%m-%d %H:%M:%S"
echo "end"