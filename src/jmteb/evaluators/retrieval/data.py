from __future__ import annotations

import json
from abc import ABC, abstractmethod

import datasets
import smart_open
from loguru import logger
from pydantic.dataclasses import dataclass
import torch.distributed


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

    def _build_idx_docid_mapping(self, dataset_attr_name: str = "dataset") -> None:
        self.idx_to_docid: dict = {}
        self.docid_to_idx: dict = {}
        id_key: str = getattr(self, "id_key", None)
        dataset = getattr(self, dataset_attr_name)
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
        # self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.query_key = query_key
        self.relevant_docs_key = relevant_docs_key
        self._build_dataset(path, split, name)

    def _build_dataset(self, path, split, name):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            self.rank_cache_path = f"/tmp/{path}_{split}_{name}.dataset"
            if rank == 0:
                self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
                self.dataset.save_to_disk(self.rank_cache_path)
                logger.critical("Loaded dataset at rank 0")
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                self.dataset = datasets.load_from_disk(self.rank_cache_path)
                logger.critical(f"Loaded dataset at rank {rank}")
        else:
            self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)

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
        self.dataset: datasets.Dataset = datasets.load_dataset("json", data_files=filename)["train"]
        self.query_key = query_key
        self.relevant_docs_key = relevant_docs_key

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
        logger.info(f"Loading dataset {path} (name={name}) with split {split}")
        self.path = path
        self.split = split
        self.name = name
        self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.id_key = id_key
        self.text_key = text_key
        self._build_idx_docid_mapping()

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
        logger.info(f"Loading dataset from {filename}")
        self.filename = filename
        with smart_open.open(filename, "r", encoding="utf-8", errors="ignore") as fin:
            corpus = [json.loads(line.strip()) for line in fin.readlines()]
        self.dataset = corpus
        self.id_key = id_key
        self.text_key = text_key
        self._build_idx_docid_mapping()

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
