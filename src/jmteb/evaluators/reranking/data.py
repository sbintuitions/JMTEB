from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import ChainMap

import datasets
import smart_open
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from loguru import logger
from pydantic.dataclasses import dataclass

from jmteb.utils.dist import build_dataset_distributed, is_main_process


@dataclass
class RerankingQuery:
    query: str
    retrieved_docs: list[str | int]
    relevance_scores: list[int]


@dataclass
class RerankingDoc:
    id: str | int
    text: str


@dataclass
class RerankingPrediction:
    query: str
    relevant_docs: list[RerankingDoc]
    reranked_relevant_docs: list[RerankingDoc]


class RerankingQueryDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> RerankingQuery:
        pass

    def __eq__(self, __value: object) -> bool:
        return False


class RerankingDocDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> RerankingDoc:
        pass

    def __eq__(self, __value: object) -> bool:
        return False

    def _build_idx_docid_mapping(
        self, dataset_attr_name: str = "dataset", distributed_state: PartialState | None = None
    ) -> None:
        dataset = getattr(self, dataset_attr_name)
        id_key: str = getattr(self, "id_key", None)

        if not distributed_state or not isinstance(distributed_state, PartialState):
            self.idx_to_docid: dict = {}
            self.docid_to_idx: dict = {}
            if id_key:
                for idx, doc_dict in enumerate(dataset):
                    self.idx_to_docid[idx] = doc_dict[id_key]
                    self.docid_to_idx[doc_dict[id_key]] = idx
            elif isinstance(dataset[0], RerankingDoc):
                for idx, doc in enumerate(dataset):
                    doc: RerankingDoc
                    self.idx_to_docid[idx] = doc.id
                    self.docid_to_idx[doc.id] = idx
            else:
                raise ValueError(f"Invalid dataset type: list[{type(dataset[0])}]")

        else:
            idx_to_docid_gather = []
            docid_to_idx_gather = []
            with distributed_state.split_between_processes([line for line in dataset]) as d:
                idx_to_docid = {}
                docid_to_idx = {}
                if id_key:
                    for idx, doc_dict in enumerate(d):
                        idx_to_docid[idx] = doc_dict[id_key]
                        docid_to_idx[doc_dict[id_key]] = idx
                elif isinstance(dataset[0], RerankingDoc):
                    for idx, doc in enumerate(d):
                        doc: RerankingDoc
                        idx_to_docid[idx] = doc.id
                        docid_to_idx[doc.id] = idx
                else:
                    raise ValueError(f"Invalid dataset type: list[{type(dataset[0])}]")
                idx_to_docid_gather.append(idx_to_docid)
                docid_to_idx_gather.append(docid_to_idx)

            idx_to_docid_gather = gather_object(idx_to_docid_gather)
            docid_to_idx_gather = gather_object(docid_to_idx_gather)

            self.idx_to_docid = dict(ChainMap(*idx_to_docid_gather))
            self.docid_to_idx = dict(ChainMap(*docid_to_idx_gather))


class HfRerankingQueryDataset(RerankingQueryDataset):
    def __init__(
        self,
        path: str,
        split: str,
        name: str | None = None,
        query_key: str = "query",
        retrieved_docs_key: str = "retrieved_docs",
        relevance_scores_key: str = "relevance_scores",
    ):
        self.path = path
        self.split = split
        self.name = name
        self.query_key = query_key
        self.retrieved_docs_key = retrieved_docs_key
        self.relevance_scores_key = relevance_scores_key
        self.dataset = build_dataset_distributed(
            datasets.load_dataset, path=path, split=split, name=name, trust_remote_code=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RerankingQuery:
        retrieved_docs = self.dataset[idx][self.retrieved_docs_key]
        relevance_scores = self.dataset[idx][self.relevance_scores_key]

        return RerankingQuery(
            query=self.dataset[idx][self.query_key], retrieved_docs=retrieved_docs, relevance_scores=relevance_scores
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "query_key", "retrieved_docs_key", "relevance_scores_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlRerankingQueryDataset(RerankingQueryDataset):
    def __init__(
        self,
        filename: str,
        query_key: str = "query",
        retrieved_docs_key: str = "retrieved_docs",
        relevance_scores_key: str = "relevance_scores",
    ):
        self.filename = filename
        self.query_key = query_key
        self.retrieved_docs_key = retrieved_docs_key
        self.relevance_scores_key = relevance_scores_key
        self.dataset = build_dataset_distributed(datasets.load_dataset, path="json", data_files=filename)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RerankingQuery:
        retrieved_docs = self.dataset[idx][self.retrieved_docs_key]
        relevance_scores = self.dataset[idx][self.relevance_scores_key]

        return RerankingQuery(
            query=self.dataset[idx][self.query_key], retrieved_docs=retrieved_docs, relevance_scores=relevance_scores
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "query_key", "retrieved_docs_key", "relevance_scores_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class HfRerankingDocDataset(RerankingDocDataset):
    def __init__(self, path: str, split: str, name: str | None = None, id_key: str = "docid", text_key: str = "text"):
        if is_main_process():
            logger.info(f"Loading dataset {path} (name={name}) with split {split}")
        self.path = path
        self.split = split
        self.name = name
        self.id_key = id_key
        self.text_key = text_key
        self.dataset = build_dataset_distributed(
            datasets.load_dataset, path=path, split=split, name=name, trust_remote_code=True
        )
        self._build_idx_docid_mapping(distributed_state=PartialState() if torch.cuda.device_count() > 1 else None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RerankingDoc:
        return RerankingDoc(id=self.dataset[idx][self.id_key], text=self.dataset[idx][self.text_key])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "id_key", "text_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlRerankingDocDataset(RerankingDocDataset):
    def __init__(self, filename: str, id_key: str = "docid", text_key: str = "text"):
        if is_main_process():
            logger.info(f"Loading dataset from {filename}")
        self.filename = filename

        def load_corpus(filename):
            with smart_open.open(filename, "r", encoding="utf-8", errors="ignore") as fin:
                corpus = [json.loads(line.strip()) for line in fin.readlines()]
            return corpus

        self.dataset = build_dataset_distributed(load_corpus, filename=filename)
        self.id_key = id_key
        self.text_key = text_key
        self._build_idx_docid_mapping(distributed_state=PartialState() if torch.cuda.device_count() > 1 else None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RerankingDoc:
        return RerankingDoc(id=self.dataset[idx][self.id_key], text=self.dataset[idx][self.text_key].strip())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "id_key", "text_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True
