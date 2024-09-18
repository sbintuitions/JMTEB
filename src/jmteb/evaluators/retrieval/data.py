from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import ChainMap

import datasets
import smart_open
import torch.distributed as dist
from accelerate import PartialState
from accelerate.utils import gather_object
from loguru import logger
from pydantic.dataclasses import dataclass

from jmteb.utils.dist import build_dataset_distributed, is_main_process


@dataclass
class RetrievalQuery:
    query: str
    relevant_docs: list[str | int]


@dataclass
class RetrievalDoc:
    id: str | int
    text: str


@dataclass
class RetrievalPrediction:
    query: str
    relevant_docs: list[RetrievalDoc]
    predicted_relevant_docs: list[RetrievalDoc]


class RetrievalQueryDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> RetrievalQuery:
        pass

    def __eq__(self, __value: object) -> bool:
        return False


class RetrievalDocDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> RetrievalDoc:
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
            elif isinstance(dataset[0], RetrievalDoc):
                for idx, doc in enumerate(dataset):
                    doc: RetrievalDoc
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
                elif isinstance(dataset[0], RetrievalDoc):
                    for idx, doc in enumerate(d):
                        doc: RetrievalDoc
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


class HfRetrievalQueryDataset(RetrievalQueryDataset):
    def __init__(
        self,
        path: str,
        split: str,
        name: str | None = None,
        query_key: str = "query",
        relevant_docs_key: str = "relevant_docs",
    ):
        self.path = path
        self.split = split
        self.name = name
        self.query_key = query_key
        self.relevant_docs_key = relevant_docs_key
        self.dataset = build_dataset_distributed(
            datasets.load_dataset, path=path, split=split, name=name, trust_remote_code=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RetrievalQuery:
        relevant_docs = self.dataset[idx][self.relevant_docs_key]
        if not isinstance(relevant_docs, list):
            relevant_docs = [relevant_docs]

        return RetrievalQuery(query=self.dataset[idx][self.query_key], relevant_docs=relevant_docs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "query_key", "retrieved_docs_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlRetrievalQueryDataset(RetrievalQueryDataset):
    def __init__(
        self,
        filename: str,
        query_key: str = "query",
        relevant_docs_key: str = "relevant_docs",
    ):
        self.filename = filename
        self.query_key = query_key
        self.relevant_docs_key = relevant_docs_key
        self.dataset = build_dataset_distributed(datasets.load_dataset, path="json", data_files=filename)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RetrievalQuery:
        relevant_docs = self.dataset[idx][self.relevant_docs_key]
        if not isinstance(relevant_docs, list):
            relevant_docs = [relevant_docs]

        return RetrievalQuery(query=self.dataset[idx][self.query_key], relevant_docs=relevant_docs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "query_key", "relevant_docs_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class HfRetrievalDocDataset(RetrievalDocDataset):
    def __init__(self, path: str, split: str, name: str | None = None, id_key: str = "docid", text_key: str = "text"):
        if is_main_process():
            logger.info(f"Loading dataset {path} (name={name}) with split {split}")
        self.path = path
        self.split = split
        self.name = name
        self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.id_key = id_key
        self.text_key = text_key
        self.dataset = build_dataset_distributed(
            datasets.load_dataset, path=path, split=split, name=name, trust_remote_code=True
        )
        self._build_idx_docid_mapping(distributed_state=PartialState() if dist.is_initialized() else None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RetrievalDoc:
        return RetrievalDoc(id=self.dataset[idx][self.id_key], text=self.dataset[idx][self.text_key])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "id_key", "text_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlRetrievalDocDataset(RetrievalDocDataset):
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
        self._build_idx_docid_mapping(distributed_state=PartialState() if dist.is_initialized() else None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RetrievalDoc:
        return RetrievalDoc(id=self.dataset[idx][self.id_key], text=self.dataset[idx][self.text_key].strip())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "id_key", "text_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True
