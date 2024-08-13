import json
import os
from pathlib import Path
from typing import Literal

import torch
import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object, find_executable_batch_size
from loguru import logger
from sentence_transformers.models import Pooling
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from jmteb.embedders.base import TextEmbedder
from jmteb.utils.dist import is_main_process


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
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
        encode_method_name: str | None = None,
        encode_method_text_argument: str = "text",
        encode_method_prefix_argument: str = "prefix",
        auto_find_batch_size: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.init_batch_size = batch_size
        if not device and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = device

        self.distributed_state = PartialState() if torch.cuda.device_count() > 1 and self.device == "cuda" else None
        if self.distributed_state and hasattr(self.distributed_state, "num_processes"):
            assert (
                self.batch_size % self.distributed_state.num_processes == 0
            ), f"""`batch_size` should be an integer multiple of the number of available GPUs,
                 but got {batch_size=}, {torch.cuda.device_count()=}. Note that `batch_size` is global batch size."""
        logger.info(f"Distribution state: {self.distributed_state}")

        model_kwargs = self._model_kwargs_parser(model_kwargs)
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True, **model_kwargs
        )
        self.dtype = getattr(self.model.config, "torch_dtype", torch.float32)

        if is_main_process():
            logger.info(f"Model loaded:\n{self.model}")
        self.normalize_embeddings = normalize_embeddings
        if self.distributed_state:
            self.model.to(self.distributed_state.device)
        else:
            self.model.to(self.device)
        logger.info(f"{self.model.device=}, {torch.cuda.device_count()=}")

        self.max_seq_length = getattr(self.model, "max_seq_length", None)
        if max_seq_length:
            self.max_seq_length = max_seq_length

        if self.max_seq_length:
            tokenizer_kwargs.update({"model_max_length": self.max_seq_length})
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, **tokenizer_kwargs
        )

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

        if self.truncate_dim:
            self.output_dim = min(self.pooling.get_sentence_embedding_dimension(), self.truncate_dim)
        else:
            self.output_dim = self.pooling.get_sentence_embedding_dimension()

        # If the network has a built-in encoding method, use it instead of `_encode`
        self.encode_method_name = encode_method_name
        self.encode_method_text_argument = encode_method_text_argument
        self.encode_method_prefix_argument = encode_method_prefix_argument

        self.auto_find_batch_size = auto_find_batch_size

    def get_output_dim(self) -> int:
        return self.output_dim

    def batch_encode_with_cache(
        self,
        text_list: list[str],
        prefix: str | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        overwrite_cache: bool = False,
        dtype: str = "float32",
    ) -> Tensor:
        if cache_path is None:
            if is_main_process():
                logger.info("Encoding embeddings")
            return self.encode(text_list, prefix=prefix).to(self._torch_dtype_parser(dtype))

        if Path(cache_path).exists() and not overwrite_cache:
            if is_main_process():
                logger.info(f"Loading embeddings from {cache_path}")
            if self.dtype is torch.bfloat16:
                dtype = torch.float16
            else:
                dtype = self.dtype
            return torch.load(cache_path).to(dtype=dtype)

        if is_main_process():
            logger.info(f"Encoding and saving embeddings to {cache_path}")

        if self.auto_find_batch_size:
            self.batch_size = self.init_batch_size

            @find_executable_batch_size(starting_batch_size=self.init_batch_size)
            def _auto_find_batch_size(batch_size, self, text_list, save_path, prefix, dtype):
                embeddings = self._batch_encode_and_save_on_disk(
                    text_list=text_list, save_path=save_path, prefix=prefix, dtype=dtype, batch_size=batch_size
                )
                self.batch_size = batch_size
                return embeddings

            return _auto_find_batch_size(self, text_list=text_list, save_path=cache_path, prefix=prefix, dtype=dtype)

        else:
            embeddings = self._batch_encode_and_save_on_disk(
                text_list, cache_path, prefix=prefix, dtype=dtype, batch_size=self.batch_size
            )

        return embeddings

    def _batch_encode_and_save_on_disk(
        self,
        text_list: list[str],
        save_path: str | os.PathLike[str],
        prefix: str | None = None,
        dtype: str = "float32",
        batch_size: int = 64,
    ) -> torch.Tensor:
        num_samples = len(text_list)
        output_dim = self.get_output_dim()
        embeddings = torch.empty((num_samples, output_dim), dtype=self._torch_dtype_parser(dtype))
        with tqdm.tqdm(total=num_samples, desc="Encoding") as pbar:
            for i in range(0, num_samples, batch_size):
                batch = text_list[i : i + batch_size]
                batch_embeddings: torch.Tensor = self.encode(batch, prefix)
                embeddings[i : i + len(batch)] = batch_embeddings.cpu()
                pbar.update(len(batch))
        torch.save(embeddings, save_path)
        return embeddings

    def encode(
        self,
        text: str | list[str],
        prefix: str | None = None,
        dtype: Literal["float32", "float16", "bfloat16"] | None = None,
    ) -> torch.Tensor:
        if self.distributed_state and len(text) >= self.distributed_state.num_processes:
            embeddings = self._encode_distributed(text, prefix)
        else:
            embeddings = self._encode(text, prefix)
        return embeddings.to(dtype=dtype, device="cpu")

    def _encode(self, text: str | list[str], prefix: str | None = None) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
            text_was_str = True
        else:
            text_was_str = False

        if self.add_eos:
            text = self._add_eos_func(text)

        if self.encode_method_name and hasattr(self.model, self.encode_method_name):
            # ensure the built-in encoding method accepts positional arguments for text and prefix
            sentence_embeddings = getattr(self.model, self.encode_method_name)(
                **{self.encode_method_text_argument: text, self.encode_method_prefix_argument: prefix}
            )
            if not isinstance(sentence_embeddings, Tensor):
                sentence_embeddings = Tensor(sentence_embeddings)

        else:
            if prefix:
                text = [prefix + t for t in text]

            encoded_input = self.tokenize(text, padding=True)
            encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}

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
                features["prompt_length"] = self.tokenize([prefix])["input_ids"].shape[-1] - 1

            # TODO: feature["token_weights_sum"]

            with torch.no_grad():
                sentence_embeddings = self.pooling.forward(features)["sentence_embedding"]

        if self.truncate_dim:
            sentence_embeddings = sentence_embeddings[..., : self.truncate_dim]
        if self.normalize_embeddings:
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        if text_was_str:
            sentence_embeddings = sentence_embeddings.view(-1)
        return sentence_embeddings

    def _encode_distributed(self, text: list[str], prefix: str | None = None) -> torch.Tensor:
        batch_gather = []
        with self.distributed_state.split_between_processes(text) as t:
            sentence_embeddings = self._encode(t, prefix)
            batch_gather.extend(torch.Tensor(sentence_embeddings).cpu())
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
            if is_main_process():
                logger.warning("No pooling config found, create a mean pooling!")
            return {"word_embedding_dimension": getattr(self.model.config, "hidden_size"), "pooling_mode": "mean"}

    def tokenize(
        self, texts: list[str] | list[dict] | list[tuple[str, str]], padding: str | bool = True
    ) -> dict[str, torch.Tensor]:
        """Tokenizes a text and maps tokens to token-ids"""
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.tokenizer.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
        )
        return output
