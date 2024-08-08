from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from accelerate.utils import find_executable_batch_size
from loguru import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from sentence_transformers.util import truncate_embeddings
from torch import Tensor
from tqdm.autonotebook import trange

from jmteb.embedders.base import TextEmbedder


class DPSentenceTransformer(SentenceTransformer):
    """SentenceBERT with pytorch torch.nn.DataParallel"""

    def __init__(self, sbert_model: SentenceTransformer):
        super(DPSentenceTransformer, self).__init__()
        self.dp_model = torch.nn.DataParallel(sbert_model)
        self.sbert = self.dp_model.module

    def forward(self, *args, **kargs):
        return self.dp_model.forward(*args, **kargs)

    def encode(
        self,
        sentences: str | list[str],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 64,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> list[Tensor] | np.ndarray | Tensor:
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.level in (logging.INFO, logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.sbert.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured "
                        f"prompts dictionary with keys {list(self.sbert.prompts.keys())!r}."
                    )
            elif self.default_prompt_name is not None:
                prompt = self.sbert.prompts.get(self.sbert.default_prompt_name, None)
        else:
            if prompt_name is not None:
                logger.warning(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features = {}
        if prompt is not None:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
            # Tracking the prompt length allow us to remove the prompt during pooling
            tokenized_prompt = self.sbert.tokenize([prompt])
            if "input_ids" in tokenized_prompt:
                extra_features["prompt_length"] = tokenized_prompt["input_ids"].shape[-1] - 1

        all_embeddings = []
        length_sorted_idx = np.argsort([-self.sbert._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.sbert.tokenize(sentences_batch)
            features.update(extra_features)

            with torch.no_grad():
                out_features = self.forward(features)

                out_features["sentence_embedding"] = truncate_embeddings(
                    out_features["sentence_embedding"], self.sbert.truncate_dim
                )

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if precision and precision != "float32":
            all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

        if convert_to_tensor:
            if len(all_embeddings):
                if isinstance(all_embeddings, np.ndarray):
                    all_embeddings = torch.from_numpy(all_embeddings)
                else:
                    all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            if not isinstance(all_embeddings, np.ndarray):
                if all_embeddings and all_embeddings[0].dtype == torch.bfloat16:
                    all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
                else:
                    all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        elif isinstance(all_embeddings, np.ndarray):
            all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


class DataParallelSentenceBertEmbedder(TextEmbedder):
    """SentenceBERT embedder with pytorch data parallel"""

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 64,
        normalize_embeddings: bool = False,
        max_seq_length: int | None = None,
        add_eos: bool = False,
        truncate_dim: int | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        auto_find_batch_size: bool = True,
    ) -> None:
        model_kwargs = self._model_kwargs_parser(model_kwargs)
        model = SentenceTransformer(
            model_name_or_path,
            trust_remote_code=True,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,  # https://github.com/UKPLab/sentence-transformers/blob/84f69fee6dcde023f46a8807e89bc99a7700ba82/sentence_transformers/SentenceTransformer.py#L81-L105  # noqa: E501
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.dp_model = DPSentenceTransformer(sbert_model=model)
        self.model = self.dp_model.sbert
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.initital_batch_size = batch_size
        self.batch_size = int(self.initital_batch_size)
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = getattr(self.model, "max_seq_length", None)
        self.add_eos = add_eos
        self.auto_find_batch_size = auto_find_batch_size

        if "torch_dtype" in model_kwargs:
            self.set_output_tensor()
        else:
            self.set_output_numpy()

    def encode(self, text: str | list[str], prefix: str | None = None) -> np.ndarray:
        if self.add_eos:
            text = self._add_eos_func(text)
        if self.auto_find_batch_size:
            # wrap function
            @find_executable_batch_size(starting_batch_size=self.batch_size)
            def _encode_with_auto_batch_size(batch_size, self, text, prefix):
                out = self.dp_model.encode(
                    text,
                    prompt=prefix,
                    convert_to_numpy=self.convert_to_numpy,
                    convert_to_tensor=self.convert_to_tensor,
                    batch_size=batch_size,
                    normalize_embeddings=self.normalize_embeddings,
                )

                self.batch_size = batch_size
                return out

            return _encode_with_auto_batch_size(self, text, prefix)
        else:
            return self.dp_model.encode(
                text,
                prompt=prefix,
                convert_to_numpy=self.convert_to_numpy,
                convert_to_tensor=self.convert_to_tensor,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
            )

    def _add_eos_func(self, text: str | list[str]) -> str | list[str]:
        try:
            eos_token = getattr(self.model.savetokenizer, "eos_token")
        except AttributeError:
            return text

        if isinstance(text, str):
            return text + eos_token
        elif isinstance(text, list):
            return [t + eos_token for t in text]

    def get_output_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
