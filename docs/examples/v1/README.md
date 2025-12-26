# Example scripts

We provide some example scripts for different scenarios.

#### [sentencebert_1gpu.sh](docs/examples/sentencebert_1gpu.sh)

For all-task evaluation with a model that can be loaded with [`SentenceTransformer`](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py) with single GPU, and `fp16` enabled. The corresponding class in `JMTEB` is [`SentenceBertEmbedder`](src/jmteb/embedders/sbert_embedder.py).

#### [sentencebert_8gpu.sh](docs/examples/sentencebert_8gpu.sh)

For all-task evaluation with a model that can be loaded with [`SentenceTransformer`](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py) with 8 GPUs in a node, and `fp16` enabled. The corresponding class in `JMTEB` is [`DataParallelSentenceBertEmbedder`](src/jmteb/embedders/data_parallel_sbert_embedder.py).

#### [transformers_embedder_multigpu.sh](docs/examples/transformers_embedder_multigpu.sh)

For all-task evaluation with a model that can be loaded with `AutoModel` in Hugging Face Transformers (even your DIY model as long as it is registered to `AutoModel`, as `trust_remote_code` is set as `True`) with 8 GPUs in a node, and `bf16` enabled. Note that to enable parallelism, `torchrun` is needed. The corresponding class in `JMTEB` is [`TransformersEmbedder`](src/jmteb/embedders/transformers_embedder.py).

#### [openai_embedder.sh](docs/examples/openai_embedder.sh)

For all-task evaluation with an OpenAI embedding model through API. Note that you must export your OpenAI API key before the evaluation. The corresponding class in `JMTEB` is [`OpenAIEmbedder`](src/jmteb/embedders/openai_embedder.py).

#### [exclude.sh](docs/examples/exclude.sh)

Exclude some slow tasks based on [sentencebert_1gpu.sh](docs/examples/sentencebert_1gpu.sh).

#### [include.sh](docs/examples/include.sh)

Specify a few tasks to be run based on [sentencebert_1gpu.sh](docs/examples/sentencebert_1gpu.sh).
