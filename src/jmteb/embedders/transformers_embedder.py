import json
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from loguru import logger
from sentence_transformers.models import Pooling
from tqdm.autonotebook import trange
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from jmteb.embedders.base import TextEmbedder


class TransformersEmbedder(TextEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        device: str | None = None,
        normalize_embeddings: bool = False,
        max_seq_length: int | None = None,
        add_eos: bool = False,
        truncate_dim: int | None = None,
        pooling_config: str | None = "1_Pooling/config.json",
        pooling_mode: str | None = None,
        tokenizer_kwargs: dict = {},
    ) -> None:
        self.model: PreTrainedModel = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.batch_size = batch_size
        if not device and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = device
        self.normalize_embeddings = normalize_embeddings

        self.distributed_state = PartialState() if torch.cuda.device_count() > 1 and self.device == "cuda" else None
        if self.distributed_state:
            self.model.to(self.distributed_state.device)
        else:
            self.model.to(self.device)
        logger.info(f"{self.model.device=}, {torch.cuda.device_count()=}")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

        self.max_seq_length = getattr(self.model, "max_seq_length", None)
        if max_seq_length:
            self.max_seq_length = max_seq_length
        self.add_eos = add_eos
        self.truncate_dim = truncate_dim

        if pooling_mode:
            pooling_config: dict = {
                "word_embedding_dimension": getattr(self.model.config, "hidden_size"),
                "pooling_mode": pooling_mode,
            }
        else:
            pooling_config: dict = self._load_pooling_config(os.path.join(model_name_or_path, pooling_config))

        self.pooling = Pooling(
            word_embedding_dimension=pooling_config.get("word_embedding_dimension"),
            pooling_mode=pooling_config.get("pooling_mode", None),
            pooling_mode_cls_token=pooling_config.get("pooling_mode_cls_token", False),
            pooling_mode_max_tokens=pooling_config.get("pooling_mode_max_tokens", False),
            pooling_mode_mean_tokens=pooling_config.get("pooling_mode_mean_tokens", False),
            pooling_mode_mean_sqrt_len_tokens=pooling_config.get("pooling_mode_mean_sqrt_len_tokens", False),
            pooling_mode_weightedmean_tokens=pooling_config.get("pooling_mode_weightedmean_tokens", False),
            pooling_mode_lasttoken=pooling_config.get("pooling_mode_lasttoken", False),
            include_prompt=pooling_config.get("include_prompt", True),
        )

        self.output_dim = self.pooling.get_sentence_embedding_dimension()

    def get_output_dim(self) -> int:
        return self.output_dim

    def encode(
        self,
        text: str | list[str],
        prefix: str | None = None,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ):
        if not convert_to_numpy ^ convert_to_tensor:
            raise ValueError("Exactly one of `convert_to_numy` and `convert_to_tensor` must be True")
        if isinstance(text, str):
            text = [text]
            text_was_str = True
        else:
            text_was_str = False

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(t) for t in text])
        text_sorted = [text[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(text), self.batch_size, desc="Batches", disable=not show_progress_bar):
            text_batch = text_sorted[start_index : start_index + self.batch_size]
            if self.distributed_state:
                batch_embeddings = self._encode_batch_distributed(text_batch, prefix)
            else:
                batch_embeddings = self._encode_batch(text_batch, prefix)
            all_embeddings.extend(batch_embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if len(all_embeddings):
            all_embeddings = torch.stack(all_embeddings)
        else:
            all_embeddings = torch.Tensor()

        if text_was_str:
            res = all_embeddings.view(-1)
        else:
            res = all_embeddings

        if convert_to_numpy:
            return res.numpy()
        else:
            return res

    def _encode_batch(self, text: list[str], prefix: str | None = None) -> torch.Tensor:
        if prefix:
            text = [prefix + t for t in text]

        if self.add_eos:
            text = self._add_eos_func(text)

        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        model_output = self.model(**encoded_input)
        last_hidden_states = model_output["last_hidden_state"]
        features = {
            "input_ids": encoded_input["input_ids"],
            "attention_mask": encoded_input["attention_mask"],
            "token_embeddings": last_hidden_states,
        }
        if "token_type_ids" in encoded_input:
            features["token_type_ids"] = encoded_input["token_type_ids"]

        if prefix:
            features["prompt_length"] = self.tokenizer([prefix], return_tensors="pt")["input_ids"].shape[-1] - 1

        # TODO: feature["token_weights_sum"]

        with torch.no_grad():
            sentence_embeddings = self.pooling.forward(features)["sentence_embedding"]
        if self.truncate_dim:
            sentence_embeddings = sentence_embeddings[..., : self.truncate_dim]
        if self.normalize_embeddings:
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def _encode_batch_distributed(self, text: list[str], prefix: str | None = None) -> torch.Tensor:
        batch_gather = []
        with self.distributed_state.split_between_processes(text) as t:
            sentence_embeddings = self._encode_batch(t, prefix)
            batch_gather.extend(sentence_embeddings.to("cpu"))

        batch_embeddings = gather_object(batch_gather)
        return torch.stack(batch_embeddings)

    def _add_eos_func(self, text: list[str]) -> list[str]:
        try:
            eos_token = getattr(self.tokenizer, "eos_token")
        except AttributeError:
            return text

        return [t + eos_token for t in text]

    def _load_pooling_config(self, config) -> dict:
        if Path(config).is_file():
            with open(Path(config), "r") as fin:
                return json.load(fin)
        else:
            logger.warning("No pooling config found, create a mean pooling!")
            return {"word_embedding_dimension": getattr(self.model.config, "hidden_size"), "pooling_mode": "mean"}
